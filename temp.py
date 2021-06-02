face_list = ['unknown', 'unknown', 'unknown', 'unknown', 'unknown',
             'alex', 'alex', 'alex', 'alex', 'alex', 'alex', 'alex', 'alex']


def check_if_safe(should_be, face_list, frame_no):
    def all_different_than(person, lst):
        for f in lst:
            if f == person:
                return False
        return True

    queue = []
    for face in face_list:
        queue.append(face)
        if len(queue) >= frame_no:
            print(queue)
            print()
            if all_different_than(should_be, queue):
                return False
            queue.pop(0)
    return True


safe = check_if_safe('alex', face_list, 5)
print(safe)
