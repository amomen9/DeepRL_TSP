"""Generate_Sample_Data.py - Generate sample TSP matrix workbooks.

Each run produces a *trio* of Excel workbooks in ``data sheets/sample data``:
    1. duration_matrix              -> generate_random_duration_matrix
    2. potential_uncertainty_matrix -> generate_random_potential_uncertainty_matrix
    3. uncertainty_inclusion_matrix -> generate_random_inclusion_matrix

Within each workbook there is one matrix per sheet; the sheet name is the
matrix dimension (e.g. "4", "5", "10", ...) written in ascending order.

The three files of a single run share:
    * a unique 3-digit id embedded in their filenames, and
    * the same generation timestamp.

Filename layout: ``<matrix_type>__<timestamp>__<id>.xlsx``.

The configured ``mat_group`` dictionaries are passed (via ``generate_data_config``)
to the three generator methods; their keys mirror the arguments of those methods.
The generation/IO logic lives in ``Library.Helper_excel.generate_sample_data``.
"""

from Library.Helper_excel import generate_sample_data


# ── Matrix-group configurations ───────────────────────────────────────────────
# Keys mirror the arguments of the three generator methods:
#   generate_random_duration_matrix(n, min_durn, max_durn, symmetric, seed)
#   generate_random_potential_uncertainty_matrix(
#       duration_matrix, min_uncertainty, max_uncertainty,
#       uncertainty_scale, uncertainty_symmetric, seed)
#   generate_random_inclusion_matrix(n, n_uncertain_routes, symmetric, seed)


matrix_groups = []

mat_group={
    "dimensions": 4,
    "duration_symmetric": True,
    "uncertainty_symmetric": False,
    "inclusion_symmetric": False,
    "n_uncertain_routes": 1,
    "min_durn": 1,
    "max_durn": 100,
    "min_uncertainty": 0.0,
    "max_uncertainty": 10.0,
    "uncertainty_scale": 0.2,
    "seed": None,
}
matrix_groups.append(mat_group)

mat_group={
    "dimensions": 5,
    "duration_symmetric": True,
    "uncertainty_symmetric": False,
    "inclusion_symmetric": False,
    "n_uncertain_routes": 2,
    "min_durn": 1,
    "max_durn": 100,
    "min_uncertainty": 0.0,
    "max_uncertainty": 10.0,
    "uncertainty_scale": 0.2,
    "seed": None,
}
matrix_groups.append(mat_group)

mat_group={
    "dimensions": 10,
    "duration_symmetric": True,
    "uncertainty_symmetric": False,
    "inclusion_symmetric": False,
    "n_uncertain_routes": 3,
    "min_durn": 1,
    "max_durn": 100,
    "min_uncertainty": 0.0,
    "max_uncertainty": 10.0,
    "uncertainty_scale": 0.2,
    "seed": None,
}
matrix_groups.append(mat_group)

mat_group={
    "dimensions": 12,
    "duration_symmetric": True,
    "uncertainty_symmetric": False,
    "inclusion_symmetric": False,
    "n_uncertain_routes": 3,
    "min_durn": 1,
    "max_durn": 100,
    "min_uncertainty": 0.0,
    "max_uncertainty": 10.0,
    "uncertainty_scale": 0.2,
    "seed": None,
}
matrix_groups.append(mat_group)

mat_group={
    "dimensions": 15,
    "duration_symmetric": True,
    "uncertainty_symmetric": False,
    "inclusion_symmetric": False,
    "n_uncertain_routes": 3,
    "min_durn": 1,
    "max_durn": 100,
    "min_uncertainty": 0.0,
    "max_uncertainty": 10.0,
    "uncertainty_scale": 0.2,
    "seed": None,
}
matrix_groups.append(mat_group)

mat_group={
    "dimensions": 20,
    "duration_symmetric": True,
    "uncertainty_symmetric": False,
    "inclusion_symmetric": False,
    "n_uncertain_routes": 5,
    "min_durn": 1,
    "max_durn": 100,
    "min_uncertainty": 0.0,
    "max_uncertainty": 10.0,
    "uncertainty_scale": 0.2,
    "seed": None,
}
matrix_groups.append(mat_group)

mat_group={
    "dimensions": 24,
    "duration_symmetric": True,
    "uncertainty_symmetric": False,
    "inclusion_symmetric": False,
    "n_uncertain_routes": 5,
    "min_durn": 1,
    "max_durn": 100,
    "min_uncertainty": 0.0,
    "max_uncertainty": 10.0,
    "uncertainty_scale": 0.2,
    "seed": None,
}
matrix_groups.append(mat_group)
mat_group={
    "dimensions": 27,
    "duration_symmetric": True,
    "uncertainty_symmetric": False,
    "inclusion_symmetric": False,
    "n_uncertain_routes": 5,
    "min_durn": 1,
    "max_durn": 100,
    "min_uncertainty": 0.0,
    "max_uncertainty": 10.0,
    "uncertainty_scale": 0.2,
    "seed": None,
}
matrix_groups.append(mat_group)
mat_group={
    "dimensions": 29,
    "duration_symmetric": True,
    "uncertainty_symmetric": False,
    "inclusion_symmetric": False,
    "n_uncertain_routes": 5,
    "min_durn": 1,
    "max_durn": 100,
    "min_uncertainty": 0.0,
    "max_uncertainty": 10.0,
    "uncertainty_scale": 0.2,
    "seed": None,
}
matrix_groups.append(mat_group)

mat_group={
    "dimensions": 30,
    "duration_symmetric": True,
    "uncertainty_symmetric": False,
    "inclusion_symmetric": False,
    "n_uncertain_routes": 5,
    "min_durn": 1,
    "max_durn": 100,
    "min_uncertainty": 0.0,
    "max_uncertainty": 10.0,
    "uncertainty_scale": 0.2,
    "seed": None,
}
matrix_groups.append(mat_group)

mat_group={
    "dimensions": 80,
    "duration_symmetric": True,
    "uncertainty_symmetric": False,
    "inclusion_symmetric": False,
    "n_uncertain_routes": 10,
    "min_durn": 1,
    "max_durn": 100,
    "min_uncertainty": 0.0,
    "max_uncertainty": 10.0,
    "uncertainty_scale": 0.2,
    "seed": None,
}
matrix_groups.append(mat_group)




generate_data_config = {
    "matrices": matrix_groups,
}


if __name__ == "__main__":
    generate_sample_data(generate_data_config)
