import csv

def load_data(filename):
    xs = []
    ys = []
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            xs.append(float(row[0]))
            ys.append(float(row[3]))
    return xs, ys

def predict(x, w0, w1):
    return (w0 + w1 * x)

def cost(xs, ys, w0, w1):
    total = 0
    for i in range(len(xs)):
        y_pred = predict(xs[i], w0, w1)
        total += (y_pred - ys[i]) ** 2
    return total / len(xs)

def train(xs, ys, learningRate = 0.00000000001, iterations = 5000):
    w0 = 0
    w1 = 0

    for i in range(iterations):
        d_w0 = 0
        d_w1 = 0

        for j in range(len(xs)):
            y_pred = predict(xs[j], w0, w1)
            error = y_pred - ys[j]
            d_w0 += error
            d_w1 += error * xs[j]

        w0 -= learningRate * d_w0
        w1 -= learningRate * d_w1
    
        if i % 500 == 0:
            print(f"Iteration {i} | Cost = {cost(xs, ys, w0, w1)} | w0 = {w0} | w1 = {w1}")

    return w0, w1

xs, ys = load_data("house_data.csv")

print("Training model...")
w0, w1 = train(xs, ys)

print("\nTraining complete!")
print("Final weights:", w0, w1)

new_x = float(input("\nEnter a house size (sq_feet) to predict price: "))
prediction = predict(new_x, w0, w1)
print(f"Predicted price: ${prediction:,.2f}")