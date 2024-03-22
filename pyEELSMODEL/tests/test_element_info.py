import os


def test_file_exists():
    filename = os.path.join('pyEELSMODEL', 'element_info.hdf5')
    assert os.path.exists(filename)


def main():
    test_file_exists()


if __name__ == "main":
    main()
