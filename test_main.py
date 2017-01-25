import main


def test_are_lines_collinear_noncollinear_lines():

    first = [0, 0, 1, 1.73]
    second = [0, 0, 1, 0.57]

    assert False == main.are_lines_collinear(first, second)


def test_are_lines_collinear_parallel_but_offset():

    first = [0, 0, 1, 1.73]
    second = [50, 0, 51, 51.73]

    assert False == main.are_lines_collinear(first, second)


def test_are_lines_collinear_collinear_lines_in_same_location():

    first = [0, 0, 1, 1.73]
    second = [0, 0, 1, 1.72]

    assert True == main.are_lines_collinear(first, second)


def test_are_lines_collinear_collinear_lines_in_different_location():

    first = [0, 0, 1, 1.73]
    second = [100, 172, 200, 344]

    assert True == main.are_lines_collinear(first, second)

