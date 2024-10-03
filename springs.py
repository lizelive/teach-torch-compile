import torch
import time

torch.set_default_device("cuda")
torch.set_default_dtype(torch.float32)


def make_connections(num_points, max_connections_per_point) -> torch.Tensor:
    """
    Returns a tensor of shape (NUM_POINTS * NUM_CONNECTIONS_PER_POINT / 2, 2) where each row
    represents a connection between two points.
    """
    connections_can = torch.cat(
        [torch.randperm(num_points) for _ in range(max_connections_per_point)]
    ).view(-1, 2)
    connections = connections_can.sort(dim=1).values.unique(dim=0)
    return connections


def make_difference_matrix(connections: torch.Tensor, num_points: int):
    assert connections.shape[1] == 2, "connections are between two points"
    num_connections = connections.shape[0]

    compressed_indices = torch.arange(0, 2 * num_connections, 2)
    plain_indices = connections.flatten()
    values = torch.tensor([1, -1], dtype=torch.float32).repeat(num_connections)

    return torch.sparse_compressed_tensor(
        compressed_indices,
        plain_indices,
        values,
        layout=torch.sparse_csr,
        size=(num_connections, num_points),
    )


def setup_for_spring_simulation(
    num_points=2**8,
    num_dims=3,
    max_connections_per_point=4,
    dense_spring_difference_matrix=False,
):
    connections = make_connections(num_points, max_connections_per_point)
    spring_difference_matrix = make_difference_matrix(connections, num_points)

    if dense_spring_difference_matrix:
        spring_difference_matrix = spring_difference_matrix.to_dense()

    point_position = torch.rand(num_points, num_dims)
    point_velocity = torch.zeros(num_points, num_dims)

    args = point_position, point_velocity

    kwargs = dict(
        # point_position=point_position,
        # point_velocity=point_velocity,
        # point_mass=torch.scalar_tensor(1.0),
        spring_target_distance=torch.scalar_tensor(1.0),
        spring_constant=torch.scalar_tensor(1.0),
        spring_difference_matrix=spring_difference_matrix,
        time_step=0.01,
    )

    return args, kwargs


def update_spring_simulation(
    point_position: torch.Tensor,
    point_velocity: torch.Tensor,
    *,
    spring_target_distance: torch.Tensor,
    spring_constant: torch.Tensor,
    spring_difference_matrix: torch.Tensor,
    time_step: float,
):
    differences = spring_difference_matrix @ point_position

    distances = torch.linalg.norm(differences, dim=1)

    spring_force = spring_constant * (distances - spring_target_distance)
    spring_acceleration = differences * (spring_force / distances).unsqueeze(1)

    point_acceleration = spring_difference_matrix.mT @ spring_acceleration

    new_point_velocity = point_velocity.add(alpha=time_step, other=point_acceleration)
    new_point_position = point_position.add(alpha=time_step, other=new_point_velocity)

    return new_point_position, new_point_velocity


class SpringSimulation(torch.nn.Module):
    spring_target_distance: torch.Tensor
    spring_constant: torch.Tensor
    spring_difference_matrix: torch.Tensor
    time_step: float

    def __init__(
        self,
        num_points=2**8,
        max_connections_per_point=4,
        dense_spring_difference_matrix=False,
        time_step=0.01,
        **kwargs
    ):
        super().__init__()

        connections = make_connections(num_points, max_connections_per_point)
        spring_difference_matrix = make_difference_matrix(connections, num_points)

        if dense_spring_difference_matrix:
            spring_difference_matrix = spring_difference_matrix.to_dense()

        self.spring_difference_matrix = spring_difference_matrix
        self.time_step = time_step
        self.spring_target_distance = torch.scalar_tensor(1.0)
        self.spring_constant = torch.scalar_tensor(1.0)

    def forward(self, point_position, point_velocity):
        differences = self.spring_difference_matrix @ point_position

        distances = torch.linalg.norm(differences, dim=1)

        spring_force = self.spring_constant * (distances - self.spring_target_distance)
        spring_acceleration = differences * (spring_force / distances).unsqueeze(1)

        point_acceleration = ( spring_acceleration.T @ self.spring_difference_matrix ).T

        new_point_velocity = point_velocity.add(
            alpha=self.time_step, other=point_acceleration
        )
        new_point_position = point_position.add(
            alpha=self.time_step, other=new_point_velocity
        )
        return new_point_position, new_point_velocity


def run_benchmark(make_simulation_fn, torch_compile_kwargs=None):
    args, kwargs = setup_for_spring_simulation()
    f = make_simulation_fn(**kwargs)

    if torch_compile_kwargs is not None:
        f = torch.compile(f, **torch_compile_kwargs)

    for phase, steps in [("warmup", 1), ("benchmark", 2**8)]:
        print(f"Phase: {phase}")

        start_time = time.time()

        for _ in range(steps):
            torch.compiler.cudagraph_mark_step_begin()
            args = f(*args)

        end_time = time.time()

        time_elapsed = end_time - start_time
        time_per_iteration = time_elapsed / steps
        print(f"time per iteration : {time_per_iteration} seconds")
        print(f"iterations per second : {1 / time_per_iteration}")


def make_binder(f):
    def _binder(**kwargs):
        def _bound(*args):
            return f(*args, **kwargs)

        return _bound

    return _binder


torch_compile_kwargs = dict(
    # backend="onnxrt",
    fullgraph=True,
    options={
        "max_autotune": True,
        "triton.cudagraphs": True,
        "trace.graph_diagram": True,
        "trace.enabled": True,
    },
)


def run_benchmarks():
    # run_benchmark(make_binder(update_spring_simulation))
    run_benchmark(SpringSimulation)
    # run_benchmark(SpringSimulation, torch_compile_kwargs)
    # run_benchmark(make_binder(update_spring_simulation), torch_compile_kwargs)


def big_dumb():
    GLOBAL_ARGS, GLOBAL_KWARGS = setup_for_spring_simulation(
        dense_spring_difference_matrix=False
    )

    spring_difference_matrix = GLOBAL_KWARGS["spring_difference_matrix"]
    time_step = GLOBAL_KWARGS["time_step"]
    spring_constant = GLOBAL_KWARGS["spring_constant"]
    spring_target_distance = GLOBAL_KWARGS["spring_target_distance"]

    @torch.compile(**torch_compile_kwargs)
    def update_spring_simulation_globals(point_position):
        differences = spring_difference_matrix @ point_position

        distances = torch.linalg.norm(differences, dim=1)

        spring_force = spring_constant * (distances - spring_target_distance)
        spring_acceleration = differences * (spring_force / distances)[:, None]

        point_acceleration = spring_difference_matrix.mT @ spring_acceleration

        return point_position + (time_step * time_step) * point_acceleration

    point_position = GLOBAL_ARGS[0]

    point_position = update_spring_simulation_globals(point_position)


def export_onnx():
    forward_args, _ = setup_for_spring_simulation()
    model = SpringSimulation()
    traced = torch.jit.script(model.eval())
    traced = torch.jit.optimize_for_inference(traced)
    print(traced.graph)
    torch.onnx.export(model, forward_args, "spring_simulation.onnx")
    model_export = torch.onnx.dynamo_export(model, *forward_args)
    model_export.save("spring_simulation.onnx")


run_benchmarks()
export_onnx()
