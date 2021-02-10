import csv

with open('NuevaBD/Abhishek_1_co.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print({", ".join(row)})
            line_count += 1
        else:
            line_count += 1
    print(f'Processed {line_count} lines.')