import random
random.seed(42)  # для воспроизводимости

print("Граф 3: Случайные веса в диапазоне от 5 до 15")
n = 5
for i in range(n):
    for j in range(n):
        if i != j:
            weight = random.randint(5, 15)
            print(f"{i} {j} {weight}")