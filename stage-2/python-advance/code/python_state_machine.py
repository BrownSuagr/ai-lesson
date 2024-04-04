def make_state_machine(status_dict):
    """
    创建一个状态机。
    :param status_dict: 包含状态名称和动作的字典。
    :return: 一个闭包，接受状态和数据作为输入。
    """
    status = None

    def state_machine(new_status, data):
        nonlocal status
        status_action = status_dict.get(status, lambda data: print(f"Invalid status: {status}"))
        new_status_action = status_dict.get(new_status, lambda data: print(f"Invalid status: {new_status}"))

        status_action(data)
        status = new_status
        new_status_action(data)

    return state_machine


# 使用例子
def print_status(data):
    print(f"Current status: {data}")


def main():
    status_machine = make_state_machine({
        'init': lambda data: print(f"Initializing {data}..."),
        'process': print_status,
        'end': lambda data: print(f"{data} processing complete.")
    })

    # 状态机运行
    status_machine('init', 'data')
    status_machine('process', 'data')
    status_machine('end', 'data')


if __name__ == '__main__':
    main()
