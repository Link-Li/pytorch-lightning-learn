

try:
    file = open("ttt", "r")
except Exception as e:
    print(e)
    # print(e.characters_written)
    print(e.args[1])
    a = 1