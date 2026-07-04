from Library.Library_env_elements import format_tour


def print_uncertainty_comparison(label, result):
    """Print deterministic and uncertain DP results side by side."""
    det = result["deterministic"]
    avg = result["average"]

    print(f"\n{'=' * 72}")
    print(f"  {label} - deterministic vs averaged uncertainty")
    print(f"{'=' * 72}")
    print(f"Deterministic cost : {det['cost']}")
    print(f"Averaged cost       : {avg['cost']}")
    print(f"Cost delta          : {avg['cost'] - det['cost']:+.6f}" if det["cost"] is not None and avg["cost"] is not None else "Cost delta          : N/A")

    print("\nDeterministic tour(s):")
    for t in det["tours"][:5]:
        print(f"  {format_tour(t)}")
    if len(det["tours"]) > 5:
        print(f"  ... and {len(det['tours']) - 5} more")

    print("\nAveraged uncertainty tour(s):")
    for t in avg["tours"][:5]:
        print(f"  {format_tour(t)}")
    if len(avg["tours"]) > 5:
        print(f"  ... and {len(avg['tours']) - 5} more")

    print("\nDP table stats:")
    print(f"  Mean table shape: {avg['dp_table_mean'].shape}")
    print(f"  Var table shape : {avg['dp_table_var'].shape}")
