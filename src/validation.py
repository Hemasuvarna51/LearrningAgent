def validate_context(context: str, objectives: list) -> bool:
    if len(context) < 50:
        return False

    score = 0
    for obj in objectives:
        if obj.split()[0].lower() in context.lower():
            score += 1

    return score >= max(1, len(objectives) // 2)
