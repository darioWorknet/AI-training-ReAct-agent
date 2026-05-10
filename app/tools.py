_users: dict[str, dict] = {
    "alice": {"name": "Alice", "address": "123 Maple St, Springfield"},
    "bob": {"name": "Bob", "address": "456 Oak Ave, Shelbyville"},
    "dario": {"name": "Dario", "address": "789 Pine Rd, Capital City"},
}


def get_user_information(name: str) -> str:
    user = _users.get(name.lower())
    if user is None:
        return f"User '{name}' not found in system"
    return f"User found: name={user['name']}, address={user['address']}"


def create_order(pizza_description: str, address: str) -> str:
    order_id = "ORD12345"
    return (
        f"Order {order_id} created for {pizza_description} to be delivered to {address}"
    )


def end_conversation() -> str:
    return "Conversation ended"
