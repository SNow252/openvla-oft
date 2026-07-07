import numpy as np


PLATE = "plate_1"
RAMEKIN = "glazed_rim_porcelain_ramekin_1"


def get_vec(obs, key):
    value = obs.get(key, None)
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32)


def dist(a, b):
    if a is None or b is None:
        return float("nan")
    return float(np.linalg.norm(a - b))


def infer_goal_for_row(row, first_obs):
    """
    Shared goal inference for task8/task1 language-control experiments.

    Returns:
        language_source, language_target, default_source, default_target

    task8 native scene:
        default_source = akita_black_bowl_1
        default_target = plate_1

        akita_black_bowl_1: bowl next to plate
        akita_black_bowl_2: bowl next to ramekin

    New mismatch conditions:
        task8_wrong_target_language_only:
            language_source = bowl1
            language_target = ramekin

        task8_wrong_source_language_only:
            language_source = bowl2
            language_target = plate

        task8_wrong_source_wrong_target_language_only:
            language_source = bowl2
            language_target = ramekin
    """
    task_id = str(row.get("task_ids", ""))
    condition = str(row.get("condition", ""))
    custom_language = str(row.get("custom_language", "")).lower()

    # Some scripts may pass dump_condition separately.
    dump_condition = str(row.get("dump_condition", ""))
    if dump_condition:
        condition = dump_condition

    # task8: original source/target template is bowl1 -> plate.
    if task_id == "8":
        default_source = "akita_black_bowl_1"
        default_target = PLATE

        source_is_ramekin_bowl = (
            condition in {
                "task8_wrong_source_language_only",
                "task8_wrong_source_wrong_target_language_only",
            }
            or "next to the ramekin" in custom_language
        )

        target_is_ramekin = (
            condition in {
                "task8_wrong_target_language_only",
                "task8_wrong_source_wrong_target_language_only",
            }
            or "place it on the ramekin" in custom_language
            or "put it on the ramekin" in custom_language
        )

        if source_is_ramekin_bowl:
            language_source = "akita_black_bowl_2"
        else:
            language_source = "akita_black_bowl_1"

        if target_is_ramekin:
            language_target = RAMEKIN
        else:
            language_target = PLATE

        return language_source, language_target, default_source, default_target

    # task1: native next-to-ramekin -> plate.
    # Infer source as the bowl closer to ramekin.
    if task_id == "1":
        bowl1_pos = get_vec(first_obs, "akita_black_bowl_1_pos")
        bowl2_pos = get_vec(first_obs, "akita_black_bowl_2_pos")
        ramekin_pos = get_vec(first_obs, f"{RAMEKIN}_pos")

        d1 = dist(bowl1_pos, ramekin_pos)
        d2 = dist(bowl2_pos, ramekin_pos)

        if d1 <= d2:
            language_source = "akita_black_bowl_1"
        else:
            language_source = "akita_black_bowl_2"

        language_target = PLATE
        default_source = language_source
        default_target = PLATE

        return language_source, language_target, default_source, default_target

    return "akita_black_bowl_1", PLATE, "akita_black_bowl_1", PLATE