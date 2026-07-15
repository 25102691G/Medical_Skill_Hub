from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any, TypeVar

from chatkit.store import NotFoundError, Store
from chatkit.types import Attachment, Page, ThreadItem, ThreadMetadata


RowT = TypeVar("RowT")


class InMemoryChatKitStore(Store[dict[str, Any]]):
    """Development store for ChatKit threads and accumulated case text."""

    def __init__(self) -> None:
        self.threads: dict[str, ThreadMetadata] = {}
        self.items: dict[str, list[ThreadItem]] = defaultdict(list)
        self.case_sections: dict[str, list[str]] = defaultdict(list)

    def append_case_section(self, thread_id: str, text: str) -> str:
        self.case_sections[thread_id].append(text)
        return self.get_case_text(thread_id)

    def get_case_text(self, thread_id: str) -> str:
        return "\n\n".join(self.case_sections.get(thread_id, []))

    def clear_case_text(self, thread_id: str) -> None:
        self.case_sections.pop(thread_id, None)

    async def load_thread(
        self,
        thread_id: str,
        context: dict[str, Any],
    ) -> ThreadMetadata:
        try:
            return self.threads[thread_id]
        except KeyError as exc:
            raise NotFoundError(f"Thread {thread_id} not found") from exc

    async def save_thread(
        self,
        thread: ThreadMetadata,
        context: dict[str, Any],
    ) -> None:
        self.threads[thread.id] = thread

    async def load_threads(
        self,
        limit: int,
        after: str | None,
        order: str,
        context: dict[str, Any],
    ) -> Page[ThreadMetadata]:
        return self._paginate(
            list(self.threads.values()),
            after,
            limit,
            order,
            sort_key=lambda thread: thread.created_at.timestamp(),
            cursor_key=lambda thread: thread.id,
        )

    async def load_thread_items(
        self,
        thread_id: str,
        after: str | None,
        limit: int,
        order: str,
        context: dict[str, Any],
    ) -> Page[ThreadItem]:
        return self._paginate(
            self.items.get(thread_id, []),
            after,
            limit,
            order,
            sort_key=lambda item: item.created_at.timestamp(),
            cursor_key=lambda item: item.id,
        )

    async def add_thread_item(
        self,
        thread_id: str,
        item: ThreadItem,
        context: dict[str, Any],
    ) -> None:
        self.items[thread_id].append(item)

    async def save_item(
        self,
        thread_id: str,
        item: ThreadItem,
        context: dict[str, Any],
    ) -> None:
        items = self.items[thread_id]
        for index, existing in enumerate(items):
            if existing.id == item.id:
                items[index] = item
                return
        items.append(item)

    async def load_item(
        self,
        thread_id: str,
        item_id: str,
        context: dict[str, Any],
    ) -> ThreadItem:
        for item in self.items.get(thread_id, []):
            if item.id == item_id:
                return item
        raise NotFoundError(f"Item {item_id} not found in thread {thread_id}")

    async def delete_thread(
        self,
        thread_id: str,
        context: dict[str, Any],
    ) -> None:
        self.threads.pop(thread_id, None)
        self.items.pop(thread_id, None)
        self.case_sections.pop(thread_id, None)

    async def delete_thread_item(
        self,
        thread_id: str,
        item_id: str,
        context: dict[str, Any],
    ) -> None:
        self.items[thread_id] = [
            item for item in self.items.get(thread_id, []) if item.id != item_id
        ]

    async def save_attachment(
        self,
        attachment: Attachment,
        context: dict[str, Any],
    ) -> None:
        raise NotImplementedError("Attachments are not enabled in this demo.")

    async def load_attachment(
        self,
        attachment_id: str,
        context: dict[str, Any],
    ) -> Attachment:
        raise NotFoundError(f"Attachment {attachment_id} not found")

    async def delete_attachment(
        self,
        attachment_id: str,
        context: dict[str, Any],
    ) -> None:
        raise NotFoundError(f"Attachment {attachment_id} not found")

    @staticmethod
    def _paginate(
        rows: list[RowT],
        after: str | None,
        limit: int,
        order: str,
        *,
        sort_key: Callable[[RowT], object],
        cursor_key: Callable[[RowT], str],
    ) -> Page[RowT]:
        sorted_rows = sorted(rows, key=sort_key, reverse=order == "desc")
        start = 0
        if after:
            for index, row in enumerate(sorted_rows):
                if cursor_key(row) == after:
                    start = index + 1
                    break

        data = sorted_rows[start : start + limit]
        has_more = start + limit < len(sorted_rows)
        next_after = cursor_key(data[-1]) if has_more and data else None
        return Page(data=data, has_more=has_more, after=next_after)
