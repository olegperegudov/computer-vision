def name_generator(size=10, chars=string.ascii_letters + string.digits):
    """This will rename all image files with random names.

    Args:
        size ([int], optional): lenght of the generated name. Defaults to 10.
        chars ([str], optional): what chars to use. Defaults to string.ascii_letters+string.digits.

    Returns:
        [str]: returns a random string of size=size
    """
    return ''.join(random.choice(chars) for _ in range(size))


for path, subdirs, files in os.walk(work_dir):
    for name in files:
        extension = name.split(".")[-1].lower()
        if extension != "jpg":
            continue

        old_name = os.path.join(path, name)
        new_name = os.path.join(path, name_generator() + "." + extension)

        os.rename(old_name, new_name)
