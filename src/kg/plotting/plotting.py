import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from kg.utils.constants import FIGURES_DIR


def find_results_files(base_dir: Path | str, allow_smoke_test: bool = False):
    """
    Collect every results.json under `base_dir`, applying only the directory
    filters (archive / bug / skip / smoke_test)
    """
    base_dir = Path(base_dir)

    results = []
    for path in base_dir.rglob("results.json"):
        parts = path.parts
        if any(sub in part for part in parts for sub in ("archive", "bug", "skip")):
            continue
        if not allow_smoke_test and any("smoke_test" in p for p in parts):
            continue
        results.append(path)

    print(f"Found {len(results)} 'results.json' files.")
    return results


def parse_path(results_file_path: Path, base_dir: Path):
    """
    Parses the file path to extract experiment metadata.
    Expected path structure relative to base_dir:
    lm_head_setting/dataset/model/patch_direction/patch_type/run_id/sentence_id/dropout_rate/results.json
    """
    # Ensure both are Path objects
    if not isinstance(results_file_path, Path):
        results_file_path = Path(results_file_path)
    if not isinstance(base_dir, Path):
        base_dir = Path(base_dir)

    try:
        # Ensure results file is within the base directory
        if base_dir not in results_file_path.parents:
            print(f"Warning: File {results_file_path} is not under base_dir {base_dir}")
            return None

        # Compute the relative path
        relative_path = results_file_path.relative_to(base_dir)
        components = list(relative_path.parts)

        if len(components) == 9 and components[-1] == "results.json":
            (
                lm_head_setting,
                dataset,
                model,
                patch_direction,
                patch_type,
                run_id,
                sentence_id,
                dropout_rate,
                _,
            ) = components

            return {
                "lm_head_setting": lm_head_setting,
                "dataset": dataset,
                "model": model,
                "patch_direction": patch_direction,
                "patch_type": patch_type,
                "run_id": run_id,
                "sentence_id": sentence_id,
                "dropout_rate": dropout_rate,
                "full_path": str(results_file_path),  # Store string path
            }
        else:
            print(
                f"Warning: Path structure mismatch for {results_file_path}. Relative: '{relative_path}', Components: {len(components)} {components}"
            )
            return None
    except Exception as e:
        print(f"Error parsing path {results_file_path}: {e}")
        return None


def calculate_metrics_from_file(results_json_path, top_k=5):
    """
    Reads a results.json file and calculates metrics.
    Metrics: mean target rank, top-k accuracy, mean target probability.
    Assumes target token rank is 1-indexed for top-k accuracy (rank <= k).
    """
    try:
        with open(results_json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading or parsing {results_json_path}: {e}")
        return None  # Indicates a file read/parse error

    if "results" not in data or not isinstance(data["results"], list):
        # print(f"Warning: 'results' key missing or not a list in {results_json_path}")
        return {  # Return NaNs if structure is invalid but file was readable
            "mean_target_rank": float("nan"),
            "top_k_accuracy": float("nan"),
            "mean_target_prob": float("nan"),
        }

    if not data["results"]:  # Empty list of results
        return {
            "mean_target_rank": float("nan"),
            "top_k_accuracy": float("nan"),
            "mean_target_prob": float("nan"),
        }

    target_ranks = []
    is_in_top_k = []
    target_probs = []

    for res_item in data["results"]:
        if "target" in res_item and isinstance(res_item["target"], dict):
            target_info = res_item["target"]

            if "token_rank" in target_info and isinstance(
                target_info["token_rank"], (int, float)
            ):
                rank = target_info["token_rank"] + 1  # Note: token_rank is 0-indexed
                target_ranks.append(rank)
                is_in_top_k.append(
                    1 if rank <= top_k and rank >= 1 else 0
                )  # Ensure rank is positive
            else:
                target_ranks.append(float("nan"))
                is_in_top_k.append(float("nan"))

            if "token_prob" in target_info and isinstance(
                target_info["token_prob"], (int, float)
            ):
                target_probs.append(target_info["token_prob"])
            else:
                target_probs.append(float("nan"))
        else:  # Target info missing for a result item
            target_ranks.append(float("nan"))
            is_in_top_k.append(float("nan"))
            target_probs.append(float("nan"))

    mean_rank = (
        np.nanmean(target_ranks)
        if any(not np.isnan(r) for r in target_ranks)
        else float("nan")
    )
    top_k_acc = (
        np.nanmean(is_in_top_k)
        if any(not np.isnan(r) for r in is_in_top_k)
        else float("nan")
    )
    mean_prob = (
        np.nanmean(target_probs)
        if any(not np.isnan(r) for r in target_probs)
        else float("nan")
    )

    return {
        "mean_target_rank": mean_rank,
        "top_k_accuracy": top_k_acc,
        "mean_target_prob": mean_prob,
    }


# pre‑compile for speed
_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")


def parse_timestamp(dir_name: str):
    """
    Extract the latest timestamp from a directory / run‑id string.
    Expected pattern: YYYY-MM-DD_HH-MM-SS (may appear multiple times).
    Returns a datetime object or None if no timestamp is found.
    """
    matches = _TS_RE.findall(dir_name)
    if not matches:
        return None
    return max(datetime.strptime(ts, "%Y-%m-%d_%H-%M-%S") for ts in matches)


def organize_results(all_results_files, base_dir: Path, top_k: int = 5):
    """
    Build nested dict:
        organized[dataset][lm_head_setting][model][sentence_id][patch] = metrics_dict
    and keep only the *newest* run for every (dataset, lm_head, model, sentence, patch) combo.
    """
    organized = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )

    parsed_ok, metrics_ok = 0, 0

    for fp in all_results_files:
        info = parse_path(fp, base_dir)
        if not info:
            continue
        parsed_ok += 1

        metrics = calculate_metrics_from_file(fp, top_k=top_k)
        if metrics is None:
            print(f"Skipping unreadable {fp}")
            continue
        metrics_ok += 1

        dset = info["dataset"]
        lm_head = info["lm_head_setting"]
        model = info["model"]
        sent = info["sentence_id"]
        patch = info["patch_type"]

        if patch == "no_patching":
            patch = (
                "no_patching_sft2pre"
                if "sft2pre" in fp.parts
                else "no_patching_pre2sft"
                if "pre2sft" in fp.parts
                else patch
            )

        ts = parse_timestamp(info["run_id"])  # newest run wins
        slot = organized[dset][lm_head][model][sent]

        if (
            patch not in slot
            or ts
            and (slot[patch]["timestamp"] is None or ts > slot[patch]["timestamp"])
        ):
            slot[patch] = {"metrics": metrics, "timestamp": ts}

    for d in organized.values():
        for l in d.values():
            for m in l.values():
                for s in m.values():
                    for p in list(s.keys()):
                        s[p] = s[p]["metrics"]

    print(f"Attempted to parse {len(all_results_files)} files.")
    print(
        f"Successfully parsed {parsed_ok} paths and calculated metrics for {metrics_ok}."
    )
    print(f"Organized data into {len(organized)} datasets.")
    return organized


# Setup order and display names for patch configs
PATCH_MAPPING = {
    "no_patching_pre2sft": ("baseline", "SFT"),
    "no_patching_sft2pre": ("baseline", "PRE"),
    "fe": ("single_token", "FE"),
    "lt": ("single_token", "LT"),
    "fe_lt": ("multi_token", "FE+LT"),
    "r": ("single_token", "R"),
    "fe_r": ("multi_token", "FE+R"),
    "r_lt": ("multi_token", "R+LT"),
    "fe_r_lt": ("multi_token", "FE+R+LT"),
    "fe_lt_complement": ("complement", "(FE+LT)^C"),
    "not_lt": ("complement", "LT^C"),
    "m": ("single_token", "M"),
    "fe_m": ("multi_token", "FE+M"),
    "fe_m_lt": ("multi_token", "FE+M+LT"),
    "m_lt": ("multi_token", "M+LT"),
    "not_fe_m": ("complement", "(FE+M)^C"),
    "not_fe_m_lt": ("complement", "(FE+M+LT)^C"),
    "not_fe": ("complement", "FE^C"),
    "attn_o_ffn": ("three_components", "ATTN+O+FFN"),
    "attn_o": ("two_components", "ATTN+O"),
    "o_ffn": ("two_components", "O+FFN"),
    "attn_ffn": ("two_components", "ATTN+FFN"),
    "o": ("one_component", "O"),
    "ffn": ("one_component", "FFN"),
    "o_ffn_up": ("ffn_comp", "O+FFN-UP"),
    "o_ffn_down": ("ffn_comp", "O+FFN-DOWN"),
    "attn_o_ffn_no_fe": ("single_token", "HYBRID (NO FE)"),
    "attn_o_ffn_full_fe": ("multi_token", "HYBRID (FE FULL)"),
    "attn_o_ffn_fe_attn": ("multi_token", "HYBRID (FE ATTN)"),
    "attn_o_ffn_fe_ffn": ("multi_token", "HYBRID (FE FFN)"),
    "attn_o_ffn_fe_o_ffn": ("multi_token", "HYBRID (FE O+FFN)"),
}

# Define the order for the buckets
BUCKET_ORDER = {
    "baseline": 0,
    "single_token": 1,
    "multi_token": 2,
    "complement": 3,
    "three_components": 4,
    "two_components": 5,
    "one_component": 6,
    "ffn_comp": 7,
}

# Skip these patch configs
SKIP_SET = {"r_rp", "r_rp_lt", "rp", "rp_lt"}

DEFAULT_BUCKET = "unknown"
DEFAULT_ORDER = 99


def get_patch_order_and_name(patch_name):
    if patch_name in PATCH_MAPPING:
        bucket, display_name = PATCH_MAPPING[patch_name]
        order = BUCKET_ORDER.get(bucket, DEFAULT_ORDER)
        return order, display_name

    return DEFAULT_ORDER, patch_name


CORE_PATCH_CONFIGS = set(
    [
        "no_patching",
        "no_patching_pre2sft",
        "no_patching_sft2pre",
        "fe",
        "lt",
        "fe_lt",
        "not_lt",
        "not_fe",
        "fe_lt_complement",
    ]
)


def plot_metric(
    organized_data,
    metric_key,
    layers_setting=None,
    save=False,
    save_dir=FIGURES_DIR,
    include_title=True,
    core_patches_only=False,
    short_title=True,
    font_size=18,
    top_k=5,
):
    """
    Generates bar plots for a specified metric across patch configurations,
    grouped by dataset, sentence, and model (in that order).

    Args:
        organized_data (dict): Nested as
            organized_data[dataset][model][sentence][patch] = metrics_dict
        metric_key (str): Metric key to plot
    """
    if not organized_data:
        print("No data available to plot.")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    metric_config = {
        "top_k_accuracy": {"label": f"Top-{top_k} Accuracy", "color": "viridis"},
        "mean_target_prob": {"label": "Mean Target Probability", "color": "plasma"},
        "mean_target_rank": {"label": "Mean Target Rank", "color": "cividis"},
    }
    if metric_key not in metric_config:
        raise ValueError(
            f"Metric '{metric_key}' is not valid. Choose from {list(metric_config.keys())}."
        )

    cfg = metric_config[metric_key]

    # Dataset → Sentence → Model
    for dataset_name, lm_head_settings in organized_data.items():
        for lm_head_setting, models_data in lm_head_settings.items():
            sentences = sorted(
                {s for m in models_data.values() for s in m}
            )  # all sentences present
            for sentence_id in sentences:
                for model_name, sentences_data in models_data.items():
                    if sentence_id not in sentences_data:
                        continue
                    patch_config_results = sentences_data[sentence_id]

                    if not patch_config_results:
                        print(
                            f"Skipping {dataset_name} / {sentence_id} / {model_name}: No patch data."
                        )
                        continue

                    patch_names, metric_values = [], []

                    # Sort first by order bucket, then alphabetically within the bucket
                    sorted_patches = sorted(
                        patch_config_results.items(),
                        key=lambda x: get_patch_order_and_name(x[0]),
                    )

                    # Collect the display names and metric values
                    seen_display_names = set()
                    for patch_name, metrics in sorted_patches:
                        if patch_name in SKIP_SET:
                            continue

                        if core_patches_only and patch_name not in CORE_PATCH_CONFIGS:
                            continue

                        if metric_key in metrics and not np.isnan(metrics[metric_key]):
                            _, display_name = get_patch_order_and_name(patch_name)

                            # Ensure uniqueness by appending index if a duplicate is found
                            if display_name in seen_display_names:
                                counter = 1
                                new_display_name = f"{display_name}_{counter}"
                                while new_display_name in seen_display_names:
                                    counter += 1
                                    new_display_name = f"{display_name}_{counter}"
                                display_name = new_display_name

                            seen_display_names.add(display_name)
                            patch_names.append(display_name)
                            metric_values.append(metrics[metric_key])

                    if not patch_names:
                        print(
                            f"No valid data for {metric_key} in {dataset_name} / {sentence_id} / {model_name}"
                        )
                        continue

                    plt.figure(figsize=(max(10, len(patch_names) * 0.8), 7))
                    colors = plt.cm.get_cmap(cfg["color"])(
                        np.linspace(0, 1, len(patch_names))
                    )
                    bars = plt.bar(patch_names, metric_values, color=colors)

                    # Define title mapping
                    metric_title_mapping = {
                        "top_k_accuracy": f"Top-{top_k} Accuracy",
                        "mean_target_prob": "Mean Target Probability",
                        "mean_target_rank": "Mean Target Rank",
                    }

                    model_title_mapping = {
                        "gpt2-xl": "GPT-2 XL",
                        "gemma": "Gemma-1.1-2B-IT",
                        "olmo": "OLMo-1B",
                        "llama3": "Llama-3.2-1B",
                        "pythia-2.8b": "Pythia-2.8B",
                        "gpt2": "GPT-2",
                    }

                    model_sentence_mapping = {
                        "sentence_1": "Sentence 1",
                        "sentence_2": "Sentence 2",
                        "sentence_3": "Sentence 3",
                        "sentence_4": "Sentence 4",
                        "counterfact_sentence": "Counterfact Sentence",
                    }

                    dataset_title_mapping = {
                        "fake_movies_real_actors": "Fake Movies, Real Actors",
                        "fake_movies_fake_actors": "Fake Movies, Fake Actors",
                        "counterfact": "Counterfact",
                        "real_movies_real_actors": "Real Movies, Real Actors",
                        "real_movies_real_actors_shuffled": "Real Movies, Real Actors (Shuffled)",
                    }

                    lm_head_title_mapping = {
                        "lm_head_always": "LM Head: Always",
                        "lm_head_never": "LM Head: Never",
                        "lm_head_last_token": "LM Head: Last Token",
                    }

                    layers_title_mapping = {
                        "all_layers": "All Layers",
                        "selective_layers": "Selective Layers",
                    }

                    if include_title and not short_title:
                        title = (
                            f"{model_title_mapping[model_name]}"
                            f" | {model_sentence_mapping[sentence_id]}"
                            f" | {dataset_title_mapping[dataset_name]}"
                            f" | {lm_head_title_mapping[lm_head_setting]}"
                            f" | {layers_title_mapping.get(layers_setting, layers_setting)}"
                        )
                        plt.title(
                            title,
                            fontsize=font_size,
                        )
                    elif include_title and short_title:
                        title = f"{model_title_mapping[model_name]}"
                        plt.figtext(
                            0.5,
                            -0.02,
                            title,
                            wrap=True,
                            horizontalalignment="center",
                            fontsize=font_size,
                        )

                    # Remove top and right spines
                    plt.gca().spines["top"].set_visible(False)
                    plt.gca().spines["right"].set_visible(False)

                    plt.ylabel(cfg["label"], fontsize=font_size)

                    # Set appropriate y-axis limits based on metric
                    if metric_key == "mean_target_rank":
                        # For rank, use a more appropriate scale
                        max_rank = max(metric_values)
                        plt.ylim(0, max_rank * 1.1)  # Add 10% padding
                    else:
                        # For probabilities and accuracy, keep 0-1 range
                        plt.ylim(0, 1.05)

                    plt.xticks(rotation=60, ha="right", fontsize=font_size)
                    plt.grid(axis="y", linestyle="--", alpha=0.7)

                    for bar in bars:
                        yval = bar.get_height()
                        plt.text(
                            bar.get_x() + bar.get_width() / 2,
                            yval,
                            f"{yval:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=font_size - 2,
                        )

                    plt.tight_layout()

                    if save:
                        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                        fname = (
                            f"{metric_key}_{dataset_name}_{sentence_id}_"
                            f"{model_name}" + f"_{lm_head_setting}" + f"_{stamp}.png"
                        )
                        plt.savefig(save_dir / fname, dpi=300, bbox_inches="tight")

                    plt.show()
                    plt.close()
