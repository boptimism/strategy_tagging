import numpy as np

# ports config is :
#        1
#     2  3  4
#  5  6  7  8  9
# define the sides of SameSide strategy
LEFT = {1: [2, 5, 6],
        2: [5],
        3: [2, 5, 6],
        4: [1, 2, 3, 5, 6, 7],
        5: [],
        6: [5],
        7: [2, 5, 6],
        8: [1, 2, 3, 5, 6, 7],
        9: [1, 2, 3, 4, 5, 6, 7, 8]
        }

RIGHT = {1: [4, 8, 9],
         2: [1, 3, 4, 7, 8, 9],
         3: [4, 8, 9],
         4: [9],
         5: [1, 2, 3, 4, 6, 7, 8, 9],
         6: [1, 3, 4, 7, 8, 9],
         7: [4, 8, 9],
         8: [9],
         9: []
         }

UP = {6: [2],
      7: [1, 3],
      8: [4],
      3: [1],
      1: [],
      2: [],
      5: [],
      4: [],
      9: []
      }

DOWN = {1: [3, 7],
        2: [6],
        3: [7],
        4: [8],
        5: [],
        6: [],
        7: [],
        8: [],
        9: []}


class AbsentAction(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return repr(self.msg)


class AbsentStrat(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return repr(self.msg)


class RandomBet(object):
    # previous trial configuration
    # pre_config = [fix_port, surebet_port, lottery_port]
    # current trial configuration
    # cur_config = [fix_port, surebet_port, lottery_port]
    # pre_delta_reward: reward(t-1)-reward(t-2), measuring
    # how reward is changed. Useful for winloseshift strategy.
    def __init__(self,
                 pre_config, cur_config,
                 pre_action, pre_reward, pre_delta_reward):
        self.pre_fixport = pre_config[0]
        self.pre_surebetport = pre_config[1]
        self.pre_lotteryport = pre_config[2]
        self.pre_action = pre_action
        self.pre_reward = pre_reward
        self.pre_delta_reward = pre_delta_reward
        self.cur_fixport = cur_config[0]
        self.cur_surebetport = cur_config[1]
        self.cur_lotteryport = cur_config[2]
        self.cur_action = -1
        self.cur_reward = -1.
        self.cur_delta_reward = -1.

    def action(self):
        # agent poke a port based on its strategy
        choice = range(1, 10)
        choice.remove(self.cur_fixport)
        self.cur_action = choice[np.random.randint(0, 8)]
        return self.cur_action

    def reward(self, lottery_prob, lottery_max_reward, surebet_reward):
        # the reward received based on action taken by agent
        if self.action < 0:
            errstr = "Run .action() first to get current port"
            raise AbsentAction(errstr)

        if self.cur_action == self.cur_lotteryport:
            # generate random number
            p = np.random.random() <= lottery_prob
            self.cur_reward = lottery_max_reward*p

        elif self.cur_action == self.cur_surebetport:
            self.cur_reward = surebet_reward

        else:
            self.cur_reward = 0

        self.cur_delta_reward = self.cur_reward - self.pre_reward

        return self.cur_reward, self.cur_delta_reward


class SameSide(RandomBet):
    def __init__(self,
                 pre_config, cur_config,
                 pre_action, pre_reward, pre_delta_reward):
        super(SameSide, self).__init__(pre_config,
                                       cur_config,
                                       pre_action,
                                       pre_reward,
                                       pre_delta_reward)

    def action(self):
        # 4 sides: left, right, up, down
        if self.pre_action in LEFT[self.pre_fixport]:
            choice = LEFT[self.cur_fixport]

        elif self.pre_action in RIGHT[self.pre_fixport]:
            choice = RIGHT[self.cur_fixport]

        elif self.pre_action in UP[self.pre_fixport]:
            choice = UP[self.cur_fixport]

        elif self.pre_action in DOWN[self.pre_fixport]:
            choice = DOWN[self.cur_fixport]

        else:
            choice = [self.cur_fixport]

        # choosing one port to poke
        if len(choice) == 0:
            self.cur_action = self.cur_fixport
        else:
            self.cur_action = choice[np.random.randint(0, len(choice))]

        return self.cur_action


class SameChoice(RandomBet):
    def __init__(self,
                 pre_config, cur_config,
                 pre_action, pre_reward, pre_delta_reward):
        super(SameChoice, self).__init__(pre_config,
                                         cur_config,
                                         pre_action,
                                         pre_reward,
                                         pre_delta_reward)

    def action(self):
        if self.pre_action == self.pre_lotteryport:
            self.cur_action = self.cur_lotteryport

        elif self.pre_action == self.pre_surebetport:
            self.cur_action = self.cur_surebetport

        else:
            self.cur_action = self.cur_fixport

        return self.cur_action


class SameAction(RandomBet):
    def __init__(self,
                 pre_config, cur_config,
                 pre_action, pre_reward, pre_delta_reward):
        super(SameAction, self).__init__(pre_config,
                                         cur_config,
                                         pre_action,
                                         pre_reward,
                                         pre_delta_reward)

    def action(self):
        self.cur_action = self.pre_action
        return self.cur_action


class Utility(RandomBet):
    def __init__(self,
                 pre_config, cur_config,
                 pre_action, pre_reward, pre_delta_reward):
        super(Utility, self).__init__(pre_config,
                                      cur_config,
                                      pre_action,
                                      pre_reward,
                                      pre_delta_reward)

    def action(self, alpha, beta, prob, max_reward, surebet_reward):
        ulottery = (max_reward*prob)**alpha
        usurebet = surebet_reward**alpha
        unothing = 0.0
        a = np.exp(beta*(usurebet - ulottery)) + \
            np.exp(beta*(unothing - ulottery))
        b = np.exp(beta*(ulottery - usurebet)) + \
            np.exp(beta*(unothing - usurebet))

        p_lottery = 1./(1. + a)
        p_surebet = 1./(1. + b)

        arandnum = np.random.random()

        if arandnum < p_lottery:
            self.cur_action = self.cur_lotteryport
        elif arandnum < p_lottery + p_surebet:
            self.cur_action = self.cur_surebetport
        else:
            ports = range(1, 10)
            ports.remove(self.cur_lotteryport)
            ports.remove(self.cur_surebetport)
            self.cur_action = ports[np.random.randint(0, 7)]
        return self.cur_action


class WinLoseShift(RandomBet):
    def __init__(self,
                 pre_config, cur_config,
                 pre_action, pre_reward, pre_delta_reward):
        super(WinLoseShift, self).__init__(pre_config,
                                           cur_config,
                                           pre_action,
                                           pre_reward,
                                           pre_delta_reward)

    def action(self, kappa):
        if self.pre_delta_reward <= 0:
            if self.pre_action == self.pre_lotteryport:
                self.cur_action = self.cur_surebetport
            elif self.pre_action == self.pre_surebetport:
                self.cur_action = self.cur_lotteryport
            else:
                self.cur_action = self.cur_fixport

        else:
            prob_stay = np.exp(-kappa*self.pre_delta_reward)
            if self.pre_action == self.pre_lotteryport:
                arandnum = np.random.random()
                if arandnum < prob_stay:
                    self.cur_action = self.cur_lotteryport
                else:
                    self.cur_action = self.cur_surebetport

            elif self.pre_action == self.pre_surebetport:
                arandnum = np.random.random()
                if arandnum < prob_stay:
                    self.cur_action = self.cur_surebetport
                else:
                    self.cur_action = self.cur_lotteryport

            else:
                self.cur_action = self.cur_fixport

        return self.cur_action
