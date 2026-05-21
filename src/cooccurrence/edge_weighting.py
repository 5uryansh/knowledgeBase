CONTEXT_WEIGHTS = {
    "sentence": 3,
    "paragraph": 2,
    "conversation": 1
}


def get_context_weight(context):
    return CONTEXT_WEIGHTS.get(context, 1)