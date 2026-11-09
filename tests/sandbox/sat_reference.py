from decimal import Decimal, localcontext


def obb_separation_exact(first, second):
    with localcontext() as context:
        context.prec = 100
        decimal = Decimal.from_float
        basis = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))

        def dot(left, right):
            return sum((a * b for a, b in zip(left, right)), Decimal(0))

        def cross(left, right):
            return (
                left[1] * right[2] - left[2] * right[1],
                left[2] * right[0] - left[0] * right[2],
                left[0] * right[1] - left[1] * right[0],
            )

        def axes(box):
            quaternion = tuple(decimal(value) for value in (
                box.orientation.w, box.orientation.x,
                box.orientation.y, box.orientation.z,
            ))
            norm = sum((component * component for component in quaternion), Decimal(0)).sqrt()
            w, x, y, z = (component / norm for component in quaternion)
            vector = (x, y, z)
            norm_squared = dot(vector, vector)
            result = []
            for raw_axis in basis:
                axis = tuple(decimal(component) for component in raw_axis)
                projection = dot(vector, axis)
                perpendicular = cross(vector, axis)
                result.append(tuple(
                    2 * vector[i] * projection
                    + axis[i] * (w * w - norm_squared)
                    + 2 * w * perpendicular[i]
                    for i in range(3)
                ))
            return tuple(result)

        first_axes = axes(first)
        second_axes = axes(second)
        first_half = tuple(decimal(value) for value in (
            first.half_extents.x, first.half_extents.y, first.half_extents.z,
        ))
        second_half = tuple(decimal(value) for value in (
            second.half_extents.x, second.half_extents.y, second.half_extents.z,
        ))
        offset = tuple(
            decimal(b) - decimal(a)
            for a, b in zip(
                (first.center.x, first.center.y, first.center.z),
                (second.center.x, second.center.y, second.center.z),
            )
        )
        gaps = []
        for raw_axis in (
            *first_axes,
            *second_axes,
            *(cross(a, b) for a in first_axes for b in second_axes),
        ):
            magnitude_squared = dot(raw_axis, raw_axis)
            if magnitude_squared == 0:
                continue
            magnitude = magnitude_squared.sqrt()
            axis = tuple(component / magnitude for component in raw_axis)
            first_radius = sum((
                extent * abs(dot(axis, unit))
                for extent, unit in zip(first_half, first_axes)
            ), Decimal(0))
            second_radius = sum((
                extent * abs(dot(axis, unit))
                for extent, unit in zip(second_half, second_axes)
            ), Decimal(0))
            gaps.append(abs(dot(offset, axis)) - first_radius - second_radius)
        return max(gaps)


def obb_separation_vertices_exact(first, second):
    """Independent SAT gap from rotation matrices and projected OBB vertices."""
    with localcontext() as context:
        context.prec = 100
        decimal = Decimal.from_float

        def dot(left, right):
            return sum((a * b for a, b in zip(left, right)), Decimal(0))

        def cross(left, right):
            return (
                left[1] * right[2] - left[2] * right[1],
                left[2] * right[0] - left[0] * right[2],
                left[0] * right[1] - left[1] * right[0],
            )

        def rotation_matrix(box):
            w, x, y, z = (decimal(value) for value in (
                box.orientation.w, box.orientation.x,
                box.orientation.y, box.orientation.z,
            ))
            norm = (w * w + x * x + y * y + z * z).sqrt()
            w, x, y, z = (value / norm for value in (w, x, y, z))
            return (
                (1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)),
                (2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)),
                (2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)),
            )

        def vertices(box, matrix):
            center = tuple(decimal(value) for value in (
                box.center.x, box.center.y, box.center.z,
            ))
            half = tuple(decimal(value) for value in (
                box.half_extents.x, box.half_extents.y, box.half_extents.z,
            ))
            axes = tuple(tuple(matrix[row][column] for row in range(3))
                         for column in range(3))
            points = []
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        points.append(tuple(
                            center[component]
                            + sx * half[0] * axes[0][component]
                            + sy * half[1] * axes[1][component]
                            + sz * half[2] * axes[2][component]
                            for component in range(3)
                        ))
            return axes, tuple(points)

        first_matrix = rotation_matrix(first)
        second_matrix = rotation_matrix(second)
        first_axes, first_vertices = vertices(first, first_matrix)
        second_axes, second_vertices = vertices(second, second_matrix)
        gaps = []
        for raw_axis in (
            *first_axes,
            *second_axes,
            *(cross(a, b) for a in first_axes for b in second_axes),
        ):
            magnitude_squared = dot(raw_axis, raw_axis)
            if magnitude_squared == 0:
                continue
            magnitude = magnitude_squared.sqrt()
            axis = tuple(value / magnitude for value in raw_axis)
            first_projection = tuple(dot(point, axis) for point in first_vertices)
            second_projection = tuple(dot(point, axis) for point in second_vertices)
            gaps.append(max(
                min(second_projection) - max(first_projection),
                min(first_projection) - max(second_projection),
            ))
        return max(gaps)
