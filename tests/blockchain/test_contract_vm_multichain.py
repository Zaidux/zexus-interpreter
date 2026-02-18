from types import SimpleNamespace

from src.zexus.blockchain.chain import Chain
from src.zexus.blockchain.contract_vm import ContractVM


def _make_dummy_contract(address: str):
    return SimpleNamespace(
        name="DummyCounter",
        address=address,
        storage=SimpleNamespace(current_state={"counter": 0}),
        actions={
            "ping": SimpleNamespace(parameters=[], body=None),
        },
    )


def test_contract_deploy_and_execute_isolated_across_chains(tmp_path):
    chain_a = Chain(chain_id="zexus-a", data_dir=str(tmp_path / "chain_a"))
    chain_b = Chain(chain_id="zexus-b", data_dir=str(tmp_path / "chain_b"))

    vm_a = ContractVM(chain_a)
    vm_b = ContractVM(chain_b)

    contract_address = "Zx01deadbeefcafe1234"
    contract_a = _make_dummy_contract(contract_address)
    contract_b = _make_dummy_contract(contract_address)

    deploy_a = vm_a.deploy_contract(contract_a, deployer="alice")
    deploy_b = vm_b.deploy_contract(contract_b, deployer="bob")

    assert deploy_a.success is True
    assert deploy_b.success is True
    assert contract_address in chain_a.contract_state
    assert contract_address in chain_b.contract_state

    # Mutate each chain independently to verify isolation
    chain_a.contract_state[contract_address]["counter"] = 11
    chain_b.contract_state[contract_address]["counter"] = 22

    exec_a = vm_a.execute_contract(contract_address, "ping", caller="alice")
    exec_b = vm_b.execute_contract(contract_address, "ping", caller="bob")

    assert exec_a.success is True
    assert exec_b.success is True
    assert chain_a.contract_state[contract_address]["counter"] == 11
    assert chain_b.contract_state[contract_address]["counter"] == 22
