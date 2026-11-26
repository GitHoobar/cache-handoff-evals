# Change index batching

Group index events four per batch. Preserve accounting batch boundaries.

The trial used the three original repository files here. The frozen model question provided a tool that changes one existing constant in `settings.py` to 4. The receiver generated its argument. The two patches reproduce the edits chosen by Qwen 3 14B when reading directly and after receiving the translated Ministral 3 14B cache. The tests here express the same two behavior checks in standard unittest form.

To inspect the results yourself, make two copies of this directory. In the first copy run `patch -p1 < native.patch`, then `python3 -B -m unittest -v`. In the second copy do the same with `handoff.patch`. The baseline also runs with `python3 -B -m unittest -v`.
