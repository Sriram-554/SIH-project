"""
SatQuery - Query Router

Classifies natural-language remote-sensing queries
into the appropriate specialist workflow.
"""

import re


def classify_query(query, number_of_images=1, modalities=None):

    query_lower = query.lower()

    if modalities is None:
        modalities = []

    # --------------------------------------------------
    # Change analysis
    # --------------------------------------------------

    change_keywords = [
        "change",
        "changed",
        "difference",
        "before and after",
        "between these dates",
        "increased",
        "decreased",
        "new construction",
        "removed"
    ]

    if (
        number_of_images >= 2
        and any(word in query_lower for word in change_keywords)
    ):
        return "change_analysis"

    # --------------------------------------------------
    # Optical + SAR
    # --------------------------------------------------

    sar_keywords = [
        "sar",
        "radar",
        "optical and sar",
        "optical + sar",
        "radar and optical"
    ]

    if any(word in query_lower for word in sar_keywords):
        return "optical_sar_analysis"

    # --------------------------------------------------
    # Region grounding
    # --------------------------------------------------

    grounding_keywords = [
        "highlight",
        "locate",
        "where is",
        "where are",
        "identify the location",
        "mark",
        "show me where"
    ]

    if any(word in query_lower for word in grounding_keywords):
        return "region_grounding"

    # --------------------------------------------------
    # Captioning / scene description
    # --------------------------------------------------

    caption_keywords = [
        "describe",
        "description",
        "scene",
        "land-cover",
        "land cover",
        "what is visible",
        "what do you see"
    ]

    if any(word in query_lower for word in caption_keywords):
        return "captioning"

    # --------------------------------------------------
    # VQA
    # --------------------------------------------------

    question_words = [
        "what",
        "which",
        "how many",
        "is there",
        "are there",
        "does",
        "do",
        "where"
    ]

    if any(
        query_lower.startswith(word)
        for word in question_words
    ):
        return "vqa"

    # --------------------------------------------------
    # Default
    # --------------------------------------------------

    return "vqa"


# ------------------------------------------------------
# Test router
# ------------------------------------------------------

if __name__ == "__main__":

    test_queries = [

        (
            "Describe the land-cover and major objects visible in this image.",
            1,
            ["optical"]
        ),

        (
            "What type of vegetation is visible?",
            1,
            ["optical"]
        ),

        (
            "Highlight the water body referred to in the query.",
            1,
            ["optical"]
        ),

        (
            "What changed between these two dates?",
            2,
            ["optical", "optical"]
        ),

        (
            "Use the optical and SAR images together to identify built-up regions.",
            2,
            ["optical", "sar"]
        ),
    ]

    print("=" * 60)
    print("SATQUERY - QUERY ROUTER TEST")
    print("=" * 60)

    for query, image_count, modalities in test_queries:

        task = classify_query(
            query,
            image_count,
            modalities
        )

        print("\nQuery:")
        print(query)

        print(f"Images    : {image_count}")
        print(f"Modalities: {modalities}")
        print(f"→ TASK    : {task}")

    print("\n" + "=" * 60)
    print("QUERY ROUTER TEST COMPLETE")
    print("=" * 60)