# for a in (0,1):
#     for b in (0,1):
#         for c in (0,1):
#             for d in (0,1):
#                 for e in (0,1):
#                     for f in (0,1):
#                         for g in (0,1):
#                             for h in (0,1):
#                                 print("{", f'{a}, {b}, {c}, {d}, {e}, {f}, {g}, {h}', "},")

# print("OUTPUT:")
# for a in (0,1):
#     for b in (0,1):
#         for c in (0,1):
#             for d in (0,1):
#                 for e in (0,1):
#                     for f in (0,1):
#                         for g in (0,1):
#                             for h in (0,1):
#                                 if h == 1:
#                                     print("{0},")
#                                 else:
#                                     print("{1},")

jumptable = [
    "{0, 0, 0},",
    "{0, 0, 1},",
    "{0, 1, 0},",
    "{0, 1, 1},",
    "{1, 0, 0},",
    "{1, 0, 1},",
    "{1, 1, 0},",
    "{1, 1, 1},",
]


for a in (0, 1):
    for b in (0, 1):
        for c in (0, 1):
            for d in (0, 1):
                for e in (0, 1):
                    for f in (0, 1):
                        for g in (0, 1):
                            print("{", f"{a}, {b}, {c}, {d}, {e}, {f}, {g}", "},")

for a in (0, 1):
    for b in (0, 1):
        for c in (0, 1):
            for d in (0, 1):
                for e in (0, 1):
                    for f in (0, 1):
                        for g in (0, 1):
                            print(
                                jumptable[a + b + c + d + e + f + g],
                            )
