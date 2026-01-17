
lines_to_add = """            except Exception as e:
                if self.env.get("_continue_on_error", False):
                    # Error Recovery Mode
                    if debug: print(f"[VM ERROR RECOVERY] {e}")
                    self.env.setdefault("_errors", []).append(str(e))
                else:
                    raise
"""

file_path = "src/zexus/vm/vm.py"
with open(file_path, "r") as f:
    lines = f.readlines()

# Find the insertion point (end of loop body)
# We know it's around line 1779
insertion_idx = -1
for i, line in enumerate(lines):
    if "self.profiler.measure_instruction(ip, elapsed)" in line:
        insertion_idx = i + 1
        break

if insertion_idx != -1:
    lines.insert(insertion_idx, lines_to_add)
    with open(file_path, "w") as f:
        f.writelines(lines)
    print("Successfully patched vm.py")
else:
    print("Could not find insertion point")
