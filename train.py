from neural import MLP

def train(model, train_xs, train_ys, iterations, grad_step_size, print_debug = False):
    for i in range(iterations):
        ypreds = [model(x) for x in train_xs]
        loss = sum((yout - ygt) ** 2 for yout, ygt in zip(ypreds, train_ys))

        if print_debug:
            print (loss)

        loss.backward()

        for p in model.parameters():
            p.data -= grad_step_size * p.grad
            p.grad = 0.0                # reset grad immediately for next iteration

def predict(model, xs):
    return [model(x) for x in xs]


model = MLP(3, [4, 4, 1])    
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]
step = 0.05

train(model, xs, ys, 200, step, True)
predictions = predict(model, xs)
print ([y for y in predictions])