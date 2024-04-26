from graphviz import Digraph

def graph(value):
    open = set()
    open.add(value)
    visited = set()
    graph = Digraph(format='svg', graph_attr={'rankdir' : 'LR'})

    while len(open):
        node = open.pop()
        if node not in visited:
            visited.add(node)
            uid = str(id(node))
            opuid = uid + node.operator
            graph.node(name=uid, label=f"{node.label} | data={node.data:.4f} | grad={node.grad:.4f}", shape='record')
            if node.operator:
                graph.node(name=opuid, label=f"{node.operator}")
                graph.edge(opuid, uid)
            for operand in node.operands:
                open.add(operand)
                graph.edge(str(id(operand)), opuid)

    return graph