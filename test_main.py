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


def test_merge_lines_simple():

    first = [0, 5, 10, 20]
    second = [20, 30, 30, 40]

    assert [0, 5, 30, 40] == main.merge_lines(first, second)


def test_merge_lines_simple_reverse_order():

    first = [0, 5, 10, 20]
    second = [20, 30, 30, 40]

    assert [0, 5, 30, 40] == main.merge_lines(second, first)


def test_merge_lines_first_inside_second():

    first = [0, 5, 10, 15]
    second = [-20, -10, 30, 60]

    assert [-20, -10, 30, 60] == main.merge_lines(first, second)


def test_merge_lines_second_inside_first():

    first = [0, 10, 100, 110]
    second = [20, 40, 30, 50]

    assert [0, 10, 100, 110] == main.merge_lines(first, second)