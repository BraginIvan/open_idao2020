from scipy.interpolate import interp1d
import numpy as np
from copy import deepcopy
from sklearn.linear_model import LinearRegression

class Extrapolator:

    def __init__(self, sat_id, point_names = ['x', 'y', 'z', 'Vx', 'Vy', 'Vz'], found_parameters=None, period = 24):
        self.period = period
        self.sat_id = sat_id
        self.point_names = point_names
        self.interpolation_size = 1000
        interpolation_step = 1 / self.interpolation_size
        self.period_size = period+1
        self.int_range = np.arange(0, self.period_size, 1.0)
        self.float_range = np.arange(0, self.period_size - 1 + interpolation_step, interpolation_step)
        self.float_range[-1] = self.period_size - 1
        if found_parameters is None:
            self.found_parameters = {}
        else:
            self.found_parameters = found_parameters

    @staticmethod
    def find_period(point_history):
        first_max = None
        second_max = None
        for i in range(1, len(point_history)-1):
            if point_history[i-1] < point_history[i] > point_history[i+1]:
                if first_max is None:
                    first_max=i
                else:
                    second_max=i
                    break
        if second_max is not None:
            return second_max-first_max
        else:
            return 24




    def _interpolated(self, circle):
        f = interp1d(self.int_range, circle)
        interpolated_circle = f(self.float_range)
        return interpolated_circle

    def get_interpolated_first_circle(self, point_history):
        first_circle = deepcopy(point_history[:self.period_size])
        interpolated_first_circle = self._interpolated(first_circle)
        # first_id=0
        # last_id =  len(interpolated_first_circle)
        #
        # first = interpolated_first_circle[first_id]
        # match = 0
        # for i in range(1, len(interpolated_first_circle)-1):
        #     if (interpolated_first_circle[i] > first > interpolated_first_circle[i+1]) or (interpolated_first_circle[i] < first < interpolated_first_circle[i+1]):
        #         match+=1
        #         if match == 2:
        #             last_id=i
        # return interpolated_first_circle[:last_id]

        return interpolated_first_circle

    def get_interpolated_last_circle(self, point_history):
        last_circle = deepcopy(point_history[-self.period_size:])
        interpolated_last_circle = self._interpolated(last_circle)
        # first_id=0
        # last = interpolated_last_circle[-1]
        #
        # match = 0
        # for i in range(len(interpolated_last_circle)-1, 0, -1):
        #     if (interpolated_last_circle[i] > last > interpolated_last_circle[i+1]) or (interpolated_last_circle[i] < last < interpolated_last_circle[i+1]):
        #         match+=1
        #         if match == 2:
        #             first_id=i
        # return interpolated_last_circle[first_id:]

        return interpolated_last_circle

    def train(self, sat_data):
        for point_name in self.point_names:
            point_history = sat_data[point_name].values
            self.found_parameters[point_name] = self.train_point(point_history)

    def get_scale_add(self, point_history):
        points_n = len(point_history)
        if len(point_history) > 100:
            minmax_range = self.period_size*2
        else:
            minmax_range = self.period_size
        first_circle_wight = point_history[:minmax_range].max() - point_history[:minmax_range].min()
        last_circle_wight = point_history[-minmax_range:].max() - point_history[-minmax_range:].min()
        scale = (last_circle_wight / first_circle_wight) - 1
        scale = scale / (points_n-self.period)

        last_mean=point_history[-self.period:].mean()
        first_mean=point_history[:self.period].mean()
        add=(last_mean-first_mean) / (points_n-self.period)
        if len(point_history) > 240:
            diffs = []
            for i in range(int(len(point_history) / 24)-1):
                start = point_history[i * self.period:(i + 1) * self.period].max() - point_history[i * self.period:(i + 1) * self.period].min()
                end = point_history[(i + 1) * self.period:(i + 2) * self.period].max() - point_history[(i + 1) * self.period:(i + 2) * self.period].min()
                diffs.append((end/start) - 1)
            diff_reg = LinearRegression().fit(np.array(range(1, int(len(point_history)/24))).reshape(-1, 1), diffs)
            scale=diff_reg.coef_[0]/self.period

            means = []
            for i in range(int(len(point_history)/24)):
                m  = (point_history[i * self.period:(i + 1) * self.period].max() + point_history[i * self.period:(i + 1) * self.period].min())/2
                means.append(m)
            reg = LinearRegression().fit(np.array(range(1, 1+int(len(point_history)/24))).reshape(-1, 1), means)



            first_mean,last_mean = reg.predict([[1],[int(len(point_history)/24)]])
            add =  reg.coef_[0]/self.period

        return scale, add, first_mean, last_mean

    def train_point(self, point_history):
        first_circle = self.get_interpolated_first_circle(point_history)
        scale, add, first_mean, last_mean = self.get_scale_add(point_history)
        best_params = {'x':0,'a': 0, 'V': 0, 'scale': scale, 'add': add, 'first_circle': first_circle, 'first_mean':first_mean, 'steps_gone': 0}
        # best_error = sum(abs(point_history - self.predict_generator(best_params, 24, len(point_history))))
        best_error = sum(abs(point_history[self.period:] - self.predict_generator(best_params, self.period, len(point_history)-self.period)))

        V_mean, a_mean,  V_var, a_var, var_reduce = self.get_initial(len(point_history))
        for _ in range(var_reduce):
            V_range = np.arange(V_mean - V_var, V_mean + V_var, V_var / 10)
            if a_var > 0:
                a_range = np.arange(a_mean - a_var, a_mean + a_var, a_var / 10)
            else:
                a_range = [0]
            for V in V_range:
                for a in a_range:
                    params = {'a': a, 'V': V, 'scale': scale, 'add': add, 'first_circle': first_circle, 'first_mean':first_mean, 'steps_gone': 0}
                    predicts = self.predict_generator(params, self.period_size, len(point_history) - self.period_size)
                    error = sum(abs(point_history[self.period_size:] - predicts))
                    if error < best_error:
                        # print(error)
                        best_error = error
                        best_params = params.copy()
            # print(best_params)
            a_mean = best_params['a']
            V_mean = best_params['V']
            a_var = a_var / 5
            V_var = V_var / 5

        last_circle = self.get_interpolated_last_circle(point_history)
        best_params['first_circle'] = last_circle
        best_params['steps_gone'] = len(point_history) - self.period_size
        best_params['first_mean'] = last_mean
        return best_params

    def predict_generator(self, point_parameters, skip_len, take_len):
        first_circle = point_parameters['first_circle']
        V = point_parameters['V']
        a = point_parameters['a']

        first_mean = point_parameters['first_mean']

        V = V * self.interpolation_size / 100
        a = a * self.interpolation_size / 100
        add = point_parameters['add']
        scale = point_parameters['scale']
        steps_gone = point_parameters['steps_gone']
        V += (steps_gone / self.period) * a
        predicts = []
        for i in range(skip_len, skip_len + take_len):
            i = i - steps_gone
            move = int(i / self.period * V + (i / self.period) ** 2 * a )
            moved = np.roll(first_circle, - move)
            calibrated = ((moved - first_mean) * (1 + scale * i)+first_mean) + i * add

            # predicts.append(calibrated[i*self.interpolation_size%len(calibrated)])

            predicts.append(calibrated[::self.interpolation_size][i % self.period])
        return predicts

    def get_initial(self, size):

        if size < 72:
            a_var = 0
            V_var = 10
            var_reduce = 2
        elif size < 240:
            a_var = 0.8
            V_var = 30
            var_reduce = 3
        elif size < 480:
            a_var = 1.2
            V_var = 50
            var_reduce = 4
        else:
            a_var = 1.2
            V_var = 80
            var_reduce = 4
        return 0, 0, V_var, a_var, var_reduce

    def eval(self, skip_len, take_len):
        return {point_name: self.predict_generator(self.found_parameters[point_name], skip_len, take_len)
                for point_name in self.point_names}
