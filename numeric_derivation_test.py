from numeric_derivation import derive


tests = [
    (lambda x: x, 2, 1),
    (lambda x: 1, 2, 0),
    (lambda x: x**2, 2, 4),
    (lambda x: x**3, 2, 12),
    (lambda x: x**4, 2, 32),
    (lambda x: x**2, 1, 2),
    (lambda x: x**3, 1, 3),
    (lambda x: 3-4*x**2, 1, -8),
    (lambda x: 3-4*x**2, 2, -16),
]


def t_derive(n):
    for f, x, expected in tests[:n]:
        actual = derive(f, x)
        assert abs(actual - expected) < .01, "Expected %s, got %s; %s" % (expected, actual, f)

def test_derive():
    t_derive(1)

def test_derive2():
    t_derive(2)

def test_derive3():
    t_derive(len(tests))

# to test the results locally, uncomment the following lines and run the file `python numeric_derivation_test.py`: 

if __name__ == "__main__":
    test_derive()
    test_derive2()
    test_derive3()