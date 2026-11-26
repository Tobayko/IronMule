from itertools import product

from training.calibration_information import bounded_dispatch, dispatch


def test_interval_veto_matches_all_corner_decisions():
    for shapes in (
        {1: [1., 1., 1., 1., 1., 1.], 2: [2., 2., 2., 2., 2., 2.]},
        {1: [1., 1., 1., 1., 1., 1.], 2: [.8, .8, .8, .8, .8, .8], 3: [.8, .8, .81, .82, .8, .8]},
        {1: [2., 2., 2., 2., 2., 2.], 2: [1., .94, .95, .96, .93, .93], 3: [1., .99, 1., 1.01, 1., 1.]},
    ):
        model = bounded_dispatch(shapes)
        keys = sorted(shapes)
        choices = [model["intervals_s"][str(k)] for k in keys]
        for t in keys:
            possible = {dispatch(dict(zip(keys, values, strict=True))).get(str(t)) for values in product(*choices)}
            assert (str(t) not in model["uncertain"]) == (possible == {model["dispatch"].get(str(t))})
