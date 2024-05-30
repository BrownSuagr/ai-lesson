import random

if __name__ == '__main__':

    student_list = [{'id': 1, 'name': 'Tom'}, {'id': 2, 'name': 'Alice'}, {'id': 3, 'name': 'Mandy'}, {'id': 4, 'name': 'Smith'}, {'id': 5, 'name': 'Amy'}, ]

    for student in student_list:
        student['age'] = random.randint(15, 20)
        student['grade'] = random.randint(60, 100)


    while True:
        print('-' * 40)
        print('欢迎使用学生管理系统')
        print('[1]添加学生信息')
        print('[2]删除学生信息')
        print('[3]查询学生信息')
        print('[9]退出系统')
        print('-' * 40)

        order = int(input('请输入功能编号：'))
        if 1 == order:
            new_student = {'id': len(student_list) + 1, 'name': input('请输入学生名称：'), 'age': int(input('请输入学生年龄：')), 'grade': int(input('请输入学生成绩：'))}
            student_list.append(new_student)
            print(f'新增成功！新增的学生信息:{new_student}')
        elif 2 == order:
            name = input('请输入需要删除的学生名称：')
            for student in student_list:
                if name == student['name']:
                    print(f'学生删除成功:{student}')
                    student_list.remove(student)
                    break
            else:
                print('学生不存在')

        elif 3 == order:
            print(f'全部学生信息:{student_list}')
        elif 9 == order:
            print('您退出了系统！')
            break
        else:
            print('输入的功能错误或者不存在！')
            continue
