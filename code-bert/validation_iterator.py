

class ValidDataIterator(object):
    def __init__(self, ds, qs, batch_size):
        self.ds = ds
        self.qs = qs
        self.total = len(ds)
        self.page = int((len(ds) +batch_size -1) / batch_size)
        self.current = 0
        self.batch_size = batch_size

    def __next__(self):
        if self.current < self.page:
            start_index = self.current * self.batch_size
            end_index = (self.current + 1) * self.batch_size

            if end_index > self.total:
                end_index = self.total
            self.current += 1

            return self.ds[start_index:end_index], self.qs[start_index:end_index]
        else:
            raise StopIteration

    def __iter__(self):
        return self


if __name__ == "__main__":
    a = [i for i in range(20)]
    it = ValidDataIterator(a, a, 10)
    for (qs, _) in it:
        print(qs)

