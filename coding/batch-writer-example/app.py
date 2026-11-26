import settings

SINK = {"index": "catalogue", "accounting": "ledger"}

def batch_rows(event, rows):
    widths = {"ledger": settings.INLINE_WIDTH, "catalogue": settings.BUFFER_WIDTH}
    n = widths[SINK[event]]
    return [rows[i:i+n] for i in range(0, len(rows), n)]
