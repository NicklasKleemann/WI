import wi_toolkit as wt

print("=== Test 1: Relational (Expecting ATIONAL -> ATE) ===")
wt.reduce_stem("relational", verbose=True)

print("\n=== Test 2: Generalizations (Expecting IZATION -> IZE) ===")
wt.reduce_stem("generalizations", verbose=True)

print("\n=== Test 3: Happy (Expecting Y -> I) ===")
wt.reduce_stem("happy", verbose=True)
