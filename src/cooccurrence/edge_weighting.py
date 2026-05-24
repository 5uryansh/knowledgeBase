CONTEXT_WEIGHTS = {
    "sentence": 5,
    "paragraph": 2
}


def get_context_weight(context):
    return CONTEXT_WEIGHTS.get(context, 1)