import pytest
import odl
import torch


@pytest.fixture(scope='session')
def size() -> int:
    return 128


@pytest.fixture(scope='session')
def angles() -> int:
    return 180


@pytest.fixture(scope='session')
def space_2D(size) -> odl.DiscretizedSpace:
    return odl.uniform_discr([-1, -1], [1, 1], [size, size])


@pytest.fixture(scope='session')
def space_3D(size) -> odl.DiscretizedSpace:
    return odl.uniform_discr([-1, -1, -1], [1, 1, 1], [size, size, size])


@pytest.fixture(scope='session')
def phantom_2D(space_2D) -> odl.DiscretizedSpaceElement:
    return odl.phantom.shepp_logan(space_2D, modified=True)


@pytest.fixture(scope='session')
def phantom_2D_tensor(phantom_2D) -> torch.Tensor:
    phantom_tensor = torch.tensor(
        phantom_2D.asarray()).unsqueeze(0)

    return torch.stack((phantom_tensor, phantom_tensor), dim=0)


@pytest.fixture(scope='session')
def phantom_3D(space_3D) -> odl.DiscretizedSpaceElement:
    return odl.phantom.shepp_logan(space_3D, modified=True)


@pytest.fixture(scope='session')
def phantom_3D_tensor(phantom_3D) -> torch.Tensor:
    phantom_tensor = torch.tensor(
        phantom_3D.asarray()).unsqueeze(0)

    return torch.stack((phantom_tensor, phantom_tensor), dim=0)
