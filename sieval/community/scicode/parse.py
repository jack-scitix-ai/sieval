# Copyright 2024 The SciCode authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Code-extraction and HDF5 target readers adapted from SciCode @ 69a8cfc:
#   - extract_function_name / get_function_from_code / process_hdf5_* :
#     src/scicode/parse/parse.py
#   - extract_python_script : src/scicode/gen/models.py
# Only the read-side helpers used by sieval are vendored; the h5 write-side
# (save_*_to_hdf5) and jsonl loaders are omitted. process_hdf5_to_tuple takes
# an explicit file path instead of the upstream module-level H5PY_FILE constant.
import ast
import re


def extract_function_name(function_header):
    pattern = r"\bdef\s+(\w+)\s*\("
    match = re.search(pattern, function_header)
    if match:
        return match.group(1)
    else:
        pattern = r"\bclass\s+(\w+)\s*\("
        match = re.search(pattern, function_header)
        if match:
            return match.group(1)
        else:
            raise ValueError("Function name or class name not found.")


def get_function_from_code(code_string, function_name):
    """Return the source of *function_name* extracted from *code_string*.

    Returns ``None`` if *code_string* is ``None``; on parse failure returns the
    original string (upstream fallback behaviour).
    """
    if code_string is None:
        return None
    try:
        tree = ast.parse(code_string)
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name == function_name
            ):
                return ast.unparse(node)
    except Exception as e:
        print(f"{function_name} not found with error: {e}")
        return code_string


def extract_python_script(response: str):
    # Extract the python script from a model response and strip import lines
    # (dependencies are prepended once from `required_dependencies`).
    if "```" in response:
        python_script = (
            response.split("```python")[1].split("```")[0]
            if "```python" in response
            else response.split("```")[1].split("```")[0]
        )
    else:
        print("Fail to extract python code from specific format.")
        python_script = response
    python_script = re.sub(
        r"^\s*(import .*|from .*\s+import\s+.*)", "", python_script, flags=re.MULTILINE
    )
    return python_script


def process_hdf5_list(group):
    lst = []
    for key in group.keys():
        lst.append(group[key][()])
    return lst


def process_hdf5_dict(group):
    import h5py  # lazy: h5py is an optional (scicode-group) dependency

    dict = {}
    for key, obj in group.items():
        if isinstance(obj, h5py.Group):
            dict[key] = process_hdf5_sparse_matrix(obj["sparse_matrix"])
        elif isinstance(obj[()], bytes):
            dict[key] = obj[()].decode("utf-8", errors="strict")
        else:
            try:
                tmp = float(key)
                dict[tmp] = obj[()]
            except ValueError:
                dict[key] = obj[()]
    return dict


def process_hdf5_sparse_matrix(group):
    import scipy.sparse  # lazy: scipy is an optional (scicode-group) dependency

    data = group["data"][()]
    shape = tuple(group["shape"][()])
    if "row" in group and "col" in group:
        row = group["row"][()]
        col = group["col"][()]
        return scipy.sparse.coo_matrix((data, (row, col)), shape=shape)
    elif "blocksize" in group:
        indices = group["indices"][()]
        indptr = group["indptr"][()]
        blocksize = tuple(group["blocksize"][()])
        return scipy.sparse.bsr_matrix(
            (data, indices, indptr), shape=shape, blocksize=blocksize
        )
    else:
        indices = group["indices"][()]
        indptr = group["indptr"][()]
        return scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)


def process_hdf5_datagroup(group):
    for key in group.keys():
        if key == "list":
            return process_hdf5_list(group[key])
        if key == "sparse_matrix":
            return process_hdf5_sparse_matrix(group[key])
        else:
            return process_hdf5_dict(group)


def process_hdf5_to_tuple(step_id, test_num, h5py_file):
    import h5py  # lazy: h5py is an optional (scicode-group) dependency

    data_lst = []
    with h5py.File(h5py_file, "r") as f:
        for test_id in range(test_num):
            group_path = f"{step_id}/test{test_id + 1}"
            if isinstance(f[group_path], h5py.Group):
                group = f[group_path]  # test1, test2, test3
                num_keys = [key for key in group.keys()]
                if len(num_keys) == 1:  # only 1 var in the test
                    subgroup = group[num_keys[0]]
                    if isinstance(subgroup, h5py.Dataset):
                        if isinstance(subgroup[()], bytes):
                            data_lst.append(subgroup[()].decode("utf-8", errors="strict"))
                        else:
                            data_lst.append(subgroup[()])
                    elif isinstance(subgroup, h5py.Group):
                        data_lst.append(process_hdf5_datagroup(subgroup))
                else:
                    var_lst = []
                    for key in group.keys():  # var1, var2, var3
                        subgroup = group[key]
                        if isinstance(subgroup, h5py.Dataset):
                            if isinstance(subgroup[()], bytes):
                                var_lst.append(
                                    subgroup[()].decode("utf-8", errors="strict")
                                )
                            else:
                                var_lst.append(subgroup[()])
                        elif isinstance(subgroup, h5py.Group):
                            var_lst.append(process_hdf5_datagroup(subgroup))
                    data_lst.append(tuple(var_lst))
            else:
                raise FileNotFoundError(f"Path {group_path} not found in the file.")
    return data_lst
