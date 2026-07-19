TEXT_COLUMN_CANDIDATES = (
    "review",
    "text",
    "tweet",
    "full_text",
    "content",
    "body",
    "message",
    "comment",
    "caption",
)


def find_text_column(columns):
    normalized = {str(column).strip().lower(): column for column in columns}
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    return None
