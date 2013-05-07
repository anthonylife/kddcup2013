import csv

class CsvDialect(csv.Dialect):
    def __init__(self):
        self.delimiter = ','
        self.doublequote = True
        self.escapechar = None
        self.lineterminator = "\n"
        self.quotechar = '"'
        self.quoting = csv.QUOTE_MINIMAL
        self.skipinitialspace = False
        self.strict = False

print "haha"
data = [row for row in csv.reader(open("/home/anthonylife/Doctor/Code/Competition/kddcup2013/data/Paper.csv"))]
print data[:15]
