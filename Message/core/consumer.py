"""Bounded asynchronous message consumers."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from bridge.context import Context, context_scope
from utils.logger_loguru import get_logger

from .handlers import MessageHandler
from .queue import queue_manager
from ..models.queue_models import MessageWrapper


logger = get_logger(__name__)


class MessageConsumer:
    """Consume a queue with a fixed number of workers.

    The previous implementation spawned one task per dequeued message.  A
    fixed worker pool makes the queue's max size a real back-pressure limit and
    keeps shutdown deterministic.
    """

    def __init__(
        self,
        queue_name: str,
        max_concurrent: int = 10,
        queue_manager_instance=None,
    ):
        self.queue_name = queue_name
        self.max_concurrent = max(1, int(max_concurrent))
        self.handlers: List[MessageHandler] = []
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        self.running = False
        self.consumer_task: Optional[asyncio.Task] = None
        self.consumer_tasks: List[asyncio.Task] = []
        self._tasks: set[asyncio.Task] = set()
        self.queue_manager = queue_manager_instance or queue_manager
        self.logger = get_logger(f"Consumer.{queue_name}")

    def add_handler(self, handler: MessageHandler) -> None:
        self.handlers.append(handler)
        self.logger.debug(f"Added handler: {handler.__class__.__name__}")

    def is_running(self) -> bool:
        return self.running

    async def start(self) -> None:
        if self.running:
            self.logger.warning(f"Consumer {self.queue_name} is already running")
            return

        self.running = True
        self.consumer_tasks = [
            asyncio.create_task(self._consume_loop(worker_id))
            for worker_id in range(self.max_concurrent)
        ]
        self.consumer_task = self.consumer_tasks[0]
        self.logger.info(f"Consumer {self.queue_name} started")

    async def _consume_loop(self, worker_id: int = 0) -> None:
        queue = self.queue_manager.get_or_create_queue(self.queue_name)
        try:
            while self.running:
                try:
                    wrapper = await queue.get(timeout=1.0)
                    if wrapper:
                        await self._process_message(wrapper)
                    elif queue.is_closed():
                        self.running = False
                        break
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.logger.error(
                        f"Consumer worker {worker_id} error: "
                        f"error_type={type(exc).__name__}"
                    )
                    await asyncio.sleep(0.1)
        finally:
            self.logger.debug(
                f"Consumer worker {worker_id} stopped: {self.queue_name}"
            )

    async def stop(self, drain_timeout: float = 5.0) -> None:
        """Stop workers cooperatively, with a bounded drain deadline."""
        self.running = False
        tasks = [task for task in self.consumer_tasks if task and not task.done()]
        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=max(0.0, float(drain_timeout)),
                )
            except asyncio.TimeoutError:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        self.consumer_tasks.clear()
        self.consumer_task = None

        pending = [task for task in self._tasks if not task.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._tasks.clear()

    async def _process_message(self, wrapper: MessageWrapper) -> None:
        async with self.semaphore:
            try:
                processed = False
                metadata = wrapper.to_metadata()
                scope = context_scope(wrapper.context)
                metadata.update(scope)
                # Preserve the legacy sender field from the trusted scope.
                metadata["from_uid"] = scope["recipient_uid"]
                # context_scope 不含 shop_name，补充写入供发送/通知使用
                try:
                    kwargs = getattr(wrapper.context, 'kwargs', None)
                    if kwargs:
                        metadata['shop_name'] = getattr(kwargs, 'shop_name', None)
                except Exception:
                    pass
                # 保留用于日志的用户键
                metadata['user_key'] = self._extract_user_id(wrapper.context)

                for handler in self.handlers:
                    try:
                        if handler.can_handle(wrapper.context):
                            success = await handler.handle(wrapper.context, metadata)
                            if success:
                                processed = True
                                self.logger.debug(
                                    f"Message {wrapper.message_id} handled by "
                                    f"{handler.__class__.__name__}"
                                )
                                break
                    except Exception as exc:
                        self.logger.error(
                            f"Handler {handler.__class__.__name__} error: "
                            f"error_type={type(exc).__name__}"
                        )
                        continue

                if not processed:
                    self.logger.warning(
                        f"Message {wrapper.message_id} not processed by any handler"
                    )
            except Exception as exc:
                self.logger.error(
                    f"Failed to process message {wrapper.message_id}: "
                    f"error_type={type(exc).__name__}"
                )

    def _extract_user_id(self, context: Context) -> str:
        scope = context_scope(context)
        return f"{scope['account_key']}|{scope['recipient_uid'] or 'unknown'}"


class MessageConsumerManager:
    """Registry for consumers belonging to one queue manager/loop."""

    def __init__(self, queue_manager_instance=None):
        self._consumers: Dict[str, MessageConsumer] = {}
        self.logger = get_logger("ConsumerManager")
        self.queue_manager = queue_manager_instance or queue_manager

    def create_consumer(
        self,
        queue_name: str,
        max_concurrent: int = 10,
    ) -> MessageConsumer:
        existing = self._consumers.get(queue_name)
        if existing is not None:
            if existing.is_running():
                self.logger.warning(f"Consumer {queue_name} already exists")
                return existing
            self._consumers.pop(queue_name, None)

        consumer = MessageConsumer(
            queue_name,
            max_concurrent,
            queue_manager_instance=self.queue_manager,
        )
        self._consumers[queue_name] = consumer
        self.logger.info(f"Created consumer: {queue_name}")
        return consumer

    def remove_consumer(self, queue_name: str) -> Optional[MessageConsumer]:
        return self._consumers.pop(queue_name, None)

    def get_consumer(self, queue_name: str) -> Optional[MessageConsumer]:
        return self._consumers.get(queue_name)

    async def start_consumer(self, queue_name: str) -> None:
        consumer = self.get_consumer(queue_name)
        if consumer:
            await consumer.start()
        else:
            self.logger.error(f"Consumer {queue_name} not found")

    async def stop_consumer(self, queue_name: str, remove: bool = False) -> None:
        consumer = self.get_consumer(queue_name)
        if consumer:
            await consumer.stop()
            if remove:
                self.remove_consumer(queue_name)
        else:
            self.logger.debug(f"Consumer {queue_name} not found")

    def list_consumers(self) -> List[str]:
        return list(self._consumers.keys())

    async def stop_all(self) -> None:
        for consumer in list(self._consumers.values()):
            await consumer.stop()
        self._consumers.clear()
        self.logger.info("All consumers stopped")


message_consumer_manager = MessageConsumerManager()
