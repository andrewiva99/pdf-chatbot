import json
from langchain_community.chat_message_histories import ChatMessageHistory


def history_to_json(memory, file_path):
    history_data = [{"role": message.type, "content": message.content} for message in memory.messages]

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(history_data, file, ensure_ascii=False, indent=4)


def history_from_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            history_data = json.load(file)

        memory = ChatMessageHistory()

        for message in history_data:
            if message['role'] == 'human':
                memory.add_user_message(message['content'])
            else:
                memory.add_ai_message(message['content'])

        return memory

    except FileNotFoundError:
        print(f"{file_path} not found")


def store_to_json(store, store_paths):
    store_ids = store.keys()
    store_paths_ids = store_paths.keys()
    for id in store_ids:
        if id in store_paths_ids:
            history_to_json(store[id], store_paths[id])


def store_from_json(store_paths):
    return {k: history_from_json(v) for k, v in store_paths.items()}


def get_chat_paths(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return {item['session_id']: item['path'] for item in data}


def save_chat_paths(store_paths, file_path):
    data = [{'session_id': k, 'path': v} for k, v in store_paths.items()]

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
