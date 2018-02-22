import visdom
import numpy as np
import os
import time

y_field_list = [ "Loss", "Acc"]
y_field_list.sort()
print(y_field_list)
smooth_len = 100


def smooth(x, smooth_len=10):
    if len(x) < smooth_len:
        return x
    box = np.ones(smooth_len) / smooth_len
    x_smooth = np.convolve(x, box, mode='same')
    for i in range(int(smooth_len / 2 + 1)):
        start = 0
        end = int(i + smooth_len / 2)
        x_smooth[i] = np.sum(x[start:end]) * 1.0 / (end - start)
    for i in range(-1, int(-smooth_len / 2), -1):
        start = int(i - smooth_len / 2)
        x_smooth[i] = np.sum(x[start:]) * 1.0 / (-start)
    return x_smooth


def parse_log(log):
    counter = 0
    # test
    with open(log, 'r') as f:
        lines = f.readlines()

    m = {}
    for field in y_field_list:
        m[field] = []
    x_list = []
    for i in range(0, len(lines)):
        line = lines[i].strip()
        if "Loss" not in line:
            continue
        # step = int(line.split('Progress: ')[1].split('/')[0].strip())

        x_list.append(counter)
        counter += 1
        line = lines[i - 1].strip()
        for field in y_field_list:
            if field in line:
                x = float(line.split(field)[1].strip().split(' ')[0])
                m[field].append(x)

    for field in y_field_list:
        m[field] = smooth(m[field], smooth_len)

    return x_list, m


class Drawing():
    def __init__(self, envs='HCCR', port=9000):
        self.vis = visdom.Visdom()
        self.vis.port = port
        self.envs = envs

    def update(self, log_info_map):
        for log_name in sorted(log_info_map.keys()):
            result_list = log_info_map[log_name]
            # envs = "{}_{}".format(self.envs, log_name.split('/')[-2])
            envs = self.envs
            win = log_name.split('/')[-1]
            # win = log_name
            print(log_name, win,len(result_list[0]),len(result_list[1]))
            self.vis.line(X=np.array(result_list[0]), Y=np.array(result_list[1]),
                          env=envs, win=win, opts=dict(title=win))


drawer = Drawing(port=12306)
while True:
    draw_info = {}
    jobnames = {
        './train.log',
    }
    for jobname in jobnames:
        log_path = jobname.split('/')[-1]
        if os.path.exists(log_path):
            print("log path= " + log_path)
            x_list, log_info = parse_log(log_path)
            # print log_info
            if len(x_list) == 0:
                continue
            for field in y_field_list:
                key = "{}/{}".format(jobname, field)
                if len(log_info[field]) == 0:
                    # print "waiting for more data"
                    continue
                if len(x_list) < smooth_len:
                    print("waiting for more data, now " + str(len(x_list)))
                    continue
                    # assert len(x_list) == len(log_info[field])
                n = min(len(x_list), len(log_info[field]))
                draw_info[key] = [x_list[:n], log_info[field][:n]]
                print("get " + field)
            print("update " + jobname)
            print(" ")
        else:
            print("Job not found")
    drawer.update(draw_info)
    print("working")
    time.sleep(10)
