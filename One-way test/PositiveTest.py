import numpy as np
import matplotlib.pyplot as plt
from array import array
from scipy import interpolate


class PositiveTest:
    def __init__(self, arr1, arr2):
        self.arr1 = arr1
        self.arr2 = arr2

    def spline_smoothing(self):
        # 样条平滑方法,平滑系数s取默认值
        spline1 = interpolate.UnivariateSpline(self.arr1, self.arr2, k=3)
        arr3 = spline1(self.arr1)
        arr4 = self.arr2 - arr3
        # 残差的标准差估计值
        sigma = arr4.std(ddof=1)
        # 判断门限
        self.lamda = 3 * sigma

    def zhenglian(self, arr_1, arr_2):
        for i in range(0, len(arr_1) - 3):
            if np.abs(np.diff(arr_2[i:i + 4], n=3)) < self.lamda:
                k = i + 3
                break
            elif i == int(len(arr_1) - 4):
                print('模型失效')
                k = i + 3
                break
        return k

    def zhengjian(self, arr_1, arr_2):
        k = self.zhenglian(arr_1, arr_2)
        result1 = np.ones(len(arr_1))
        list1 = array('d', [])
        # 这部分未被检测，默认为正常
        if k != 3:
            result1[arr_1[0]:k - 3] = 3
        # 正向递推
        csth = np.copy(arr_1[k - 3:k + 1]).tolist()
        csnh = np.copy(arr_2[k - 3:k + 1]).tolist()
        for i in range(k + 1, len(arr_1), 1):
            if np.abs(interpolate.UnivariateSpline(csth, csnh, k=3, s=0)(arr_1[i]) - arr_2[i]) < self.lamda:
                result1[i] = 1
                del csth[0]
                del csnh[0]
                csth.append(arr_1[i])
                csnh.append(arr_2[i])
            elif np.abs(interpolate.UnivariateSpline(csth, csnh, k=2, s=0)(arr_1[i]) - arr_2[i]) < self.lamda:
                result1[i] = 1
                del csth[0]
                del csnh[0]
                csth.append(arr_1[i])
                csnh.append(arr_2[i])
            elif np.abs(interpolate.UnivariateSpline(csth, csnh, k=1, s=1)(arr_1[i]) - arr_2[i]) < self.lamda:
                result1[i] = 1
                del csth[0]
                del csnh[0]
                csth.append(arr_1[i])
                csnh.append(arr_2[i])
            else:
                result1[i] = 0
                list1.append(i)
                if len(list1) >= 10:
                    # 连续异常数据超限，模型失效，需要数据分段
                    if np.sum(np.diff(list1[-10:])) == 9:
                        f = int(list1[-10])
                        print("连续异常超限")
                        # 采用递归进行分段处理，用np.hstack()横向连接
                        return np.hstack((self.zhengjian(arr_1[0:f], arr_2[0:f]), self.zhengjian(arr_1[f:], arr_2[f:])))
                        break
        return result1


def main():
    data = np.loadtxt("test_data.txt")
    # 时间序列（有缺失）
    arr1 = data[0, :]
    arr2 = data[1, :]
    eliminating = PositiveTest(arr1, arr2)
    eliminating.spline_smoothing()
    result = eliminating.zhengjian(arr1, arr2)
    # 异常数据剔除后的效果图
    arr_1 = arr1[np.nonzero(result)[0]]
    arr_2 = arr2[np.nonzero(result)[0]]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(arr_1, arr_2, 'r-')
    ax1.set_xlabel('t/s')
    ax1.set_ylabel('Vx/(m/s)')
    # 异常数据剔除后的残差图
    spline1 = interpolate.UnivariateSpline(arr_1, arr_2, k=3)
    arr3 = spline1(arr_1)
    arr4 = arr_2 - arr3
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(arr_1, arr4, 'r-')
    ax2.set_xlabel('t/s')
    ax2.set_ylabel(r"$\sigma(Vx)/(m/s)$")
    plt.yticks([-20, 0, 20])
    plt.show()

if __name__ == "__main__":
    main()
