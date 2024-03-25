w = 1.0

def forward(x):
    return x * w


def loss(xs, ys):
    y_pred = forward(x)
    return (y_pred - ys) ** 2


def gradient(xs, ys):
    return 2 * xs * (xs * w - ys)


print('Predict (before training)', 4, forward(4))

epoch_list = []
loss_list = []

for epoch in range(100):
    for x, y in zip(X_data, Y_data):
        grad_val = gradient(x, y)
        w -= 0.01 * grad_val
        print('\tgrad:', x, y, grad_val)
        l = loss(x, y)

    print('progress:', epoch, 'w=', w, 'loss=', l)
    epoch_list.append(epoch)
    loss_list.append(l)

print('Predict (after training)', 4, forward(4))

plt.plot(epoch_list, loss_list)
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.show()
