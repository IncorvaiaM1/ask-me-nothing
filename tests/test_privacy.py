from brokerdesk.privacy import Field, Tier, canary, find_leaks


def _field(client, name, tier, seed=0):
    c = canary(client, name, seed)
    return Field(name=name, value=f"{name}-{c}", tier=tier, client_id=client, canary=c)


def test_canary_is_deterministic_and_unique():
    assert canary("c1", "income", 0) == canary("c1", "income", 0)
    assert canary("c1", "income", 0) != canary("c2", "income", 0)
    assert canary("c1", "income", 0) != canary("c1", "income", 1)


def test_leak_of_other_clients_data_is_caught():
    mine = _field("c1", "risk_tolerance", Tier.LOW)
    theirs = _field("c2", "income", Tier.SENSITIVE)
    text = f"Your risk tolerance is {mine.value}; client 2 earns {theirs.value}."
    leaks = find_leaks(text, [mine, theirs], allowed_client="c1")
    assert leaks == [theirs]


def test_restricted_field_counts_even_for_own_client():
    ssn = _field("c1", "ssn", Tier.RESTRICTED)
    assert find_leaks(f"Confirming SSN {ssn.value}", [ssn], allowed_client="c1") == [ssn]
