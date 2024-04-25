import turtle
def tree(branchLen,t):
    if branchLen > 5:
        t.backward(branchLen)
        t.right(20)
        tree(branchLen-15,t)
        t.left(20)
        tree(branchLen-15,t)
        t.left(20)
        tree(branchLen-15,t)
        t.right(20)
        t.forward(branchLen)

def main():
    t = turtle.Turtle()
    myWin = turtle.Screen()
    t.left(90)
    t.up()
    t.forward(100)
    t.down()
    t.color("brown")
    tree(45,t)
    myWin.exitonclick()

main()
