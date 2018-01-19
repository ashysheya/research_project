import numpy as np
from skimage.draw import circle, circle_perimeter


class Circles:
    def __init__(self,
                 radius,
                 bg_color,
                 perim_color,
                 circle_color,
                 size=(128, 128),
                 n_obj=10,
                 var_n_obj=5,
                 var_size=2,
                 maxover=-6,
                 noise_level=0.05,
                 num_tries=15):
        self.bg_color = bg_color
        self.perim_color = perim_color
        self.circle_color = circle_color
        self.size = size
        self.radius = radius
        self.max_overlap = maxover
        self.noise_level = noise_level
        self.n_obj = n_obj
        self.var_n_obj = var_n_obj
        self.var_size = var_size
        self.num_tries = num_tries

    @staticmethod
    def __dist__(a, b):
        x1, y1, r1 = a
        x2, y2, r2 = b
        r = r1 + r2
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) - r

    def get_valid_batch(self, batch_size=40, transforms=[]):
        return self.get_train_batch(batch_size, transforms)

    def get_train_batch(self, batch_size=40, transforms=[]):
        sz = self.size
        pics = np.zeros((batch_size, 1, sz[0], sz[1]), dtype=np.float32)
        pics[:, :, :, :] = self.bg_color

        labels = np.zeros((batch_size, sz[0], sz[1]), dtype=np.int)
        number_of_objects = np.random.randint(self.n_obj - self.var_n_obj,
                                              self.n_obj + self.var_n_obj,
                                              batch_size)

        outline = self.radius + self.var_size

        for i in range(batch_size):
            circles = []
            for idx_circle in range(number_of_objects[i]):
                for _ in range(self.num_tries):

                    # generate new circle
                    x = np.random.randint(outline, sz[0] - outline)
                    y = np.random.randint(outline, sz[1] - outline)
                    r = np.random.randint(self.radius - self.var_size,
                                          self.radius + self.var_size)

                    mindist = None

                    # distance to other circles
                    for cur_circle in circles:
                        distance = self.__dist__((x, y, r), cur_circle)
                        if mindist is None or distance < mindist:
                            mindist = distance

                    # if valid circle
                    if mindist is None or mindist > self.max_overlap:
                        circles.append((x, y, r))
                        rr_o, cc_o = circle(x, y, r, shape=sz)
                        if len(rr_o) > 0:
                            labels[i, rr_o, cc_o] = 1
                            pics[i, 0, rr_o, cc_o] = self.circle_color
                            rr_p, cc_p = circle_perimeter(x, y, r, shape=sz)
                            labels[i, rr_p, cc_p] = 2
                            pics[i, 0, rr_p, cc_p] = self.perim_color
                            break

        noise = np.random.randn(batch_size, 1, sz[0], sz[1])
        pics += noise * self.noise_level

        return pics, labels
