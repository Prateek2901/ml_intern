import build as b

df = b.load_data()

print(df)

print(b.first_country(df))

print(b.gold_medal(df))

print(b.biggest_difference_in_gold_medal(df))

p = b.get_points(df)
print(p)

print(b.k_means(df))