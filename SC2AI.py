import sc2
from sc2 import run_game, maps, Race, Difficulty, position, Result
from sc2.player import Bot, Computer
from sc2.constants import EFFECT_VOIDRAYPRISMATICALIGNMENT, BuffId, AbilityId, UnitTypeId, ASSIMILATOR, EFFECT_CHRONOBOOSTENERGYCOST, PYLON, CYBERNETICSCORE, OBSERVER, ZEALOT, STARGATE, GATEWAY, NEXUS, VOIDRAY, PROBE, ROBOTICSFACILITY
from sc2.position import Point2, Point3
import random
import cv2
import time
import numpy as np
import math

HEADLESS = False

class SentdeBot(sc2.BotAI):
    def __init__(self):
        #self.ITERATIONS_PER_MINUTE = 165
        self.MAX_WORKERS = 80
        self.do_something_after = 0
        self.scouts_and_spots = {}
        self.train_data = []
#        self.choices = {0: self.build_scout,
#                1: self.build_zealot,
#                2: self.build_gateway,
#                3: self.build_voidray,
#                4: self.build_stalker,
#                5: self.build_worker,
#                6: self.build_assimilator,
#                7: self.build_stargate,
#                8: self.build_pylon,
#                9: self.defend_nexus,
#                10: self.attack_known_enemy_unit,
#                11: self.attack_known_enemy_structure,
#                12: self.expand,
#                13: self.do_nothing,
#                }

    def on_end(self, game_result):
        print('--- on_end called ---')
        print(game_result)

        if game_result == Result.Victory:
            np.save("train_data/{}.npy".format(str(int(time.time()))), np.array(self.train_data))



    async def on_step(self, iteration):
        #self.iteration = iteration
        ###########################
        self.times = (self.state.game_loop/22.4) / 60
        print('Time:',self.times)
        ###########################
        await self.build_scout()
        await self.scout()
        await self.distribute_workers()
        await self.build_workers()
        await self.build_pylons()
        await self.build_assimilators()
        await self.expand()
        await self.offensive_force_buildings()
        await self.build_offensive_force()
        await self.intel()
        await self.attack()
        await self.activate_voidrays()
#        await self.do_something()

    def random_location_variance(self, enemy_start_location):
        x = enemy_start_location[0]
        y = enemy_start_location[1]

        x += random.randrange(-5,5)
        y += random.randrange(-5,5)

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.game_info.map_size[0]:
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            y = self.game_info.map_size[1]

        go_to = position.Point2(position.Pointlike((x,y)))
        return go_to


    async def scout(self):
        self.expand_dis_dir = {}
        for el in self.expansion_locations:
            distance_to_enemy_start = el.distance_to(self.enemy_start_locations[0])
            self.expand_dis_dir[distance_to_enemy_start] = el
        self.ordered_exp_distances = sorted(k for k in self.expand_dis_dir)
        
        existing_ids = [unit.tag for unit in self.units]
        to_be_removed = []
        for noted_scout in self.scouts_and_spots:
            if noted_scout not in existing_ids:
                to_be_removed.append(noted_scout)
                
        for scout in to_be_removed:
            del self.scouts_and_spots[scout]
        
        if len(self.units(ROBOTICSFACILITY).ready) == 0 and self.times > 1:
            unit_type = PROBE
            unit_limit = 1
        else:
            unit_type = OBSERVER
            unit_limit = 5
        assign_scout = True
        
        if unit_type == PROBE:
            for unit in self.units(PROBE):
                if unit.tag in self.scouts_and_spots:
                    assign_scout = False
                    
        if assign_scout:
            if len(self.units(unit_type).idle) > 0:
                for obs in self.units(unit_type).idle[:unit_limit]:
                    if obs.tag not in self.scouts_and_spots:
                        for dist in self.ordered_exp_distances:
                            try:
                                location = self.expand_dis_dir[dist] #next(value for key, value in self.expand_dis_dir.items if key == dist)
                                active_locations = [self.scouts_and_spots[k] for k in self.scouts_and_spots]
                                
                                if location not in active_locations:
                                    if unit_type == PROBE:
                                        for unit in self.units(PROBE):
                                            if unit.tag in self.scouts_and_spots:
                                                continue
                                    await self.do(obs.move(location))
                                    self.scouts_and_spots[obs.tag] = location
                                    break
                            except Exception as e:
                                
                                pass
        for obs in self.units(unit_type):
            if obs.tag in self.scouts_and_spots:
                if obs in [probe for probe in self.units(PROBE)]:
                    await self.do(obs.move(self.random_location_variance(self.scouts_and_spots[obs.tag])))

    async def build_scout(self):
        if len(self.units(OBSERVER)) < math.floor(self.times/3):
            for rf in self.units(ROBOTICSFACILITY).ready.noqueue:
                print(len(self.units(OBSERVER)), self.times/3)
                if self.can_afford(OBSERVER) and self.supply_left > 0:
                    await self.do(rf.train(OBSERVER))
                    
    async def intel(self):
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)

        # UNIT: [SIZE, (BGR COLOR)]
        '''from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
 CYBERNETICSCORE, STARGATE, VOIDRAY'''
        draw_dict = {
                     NEXUS: [15, (0, 255, 0)],
                     PYLON: [3, (20, 235, 0)],
                     PROBE: [1, (55, 200, 0)],
                     ASSIMILATOR: [2, (55, 200, 0)],
                     GATEWAY: [3, (200, 100, 0)],
                     CYBERNETICSCORE: [3, (150, 150, 0)],
                     STARGATE: [5, (255, 0, 0)],
                     ROBOTICSFACILITY: [5, (215, 155, 0)],

                     VOIDRAY: [3, (255, 100, 0)],
                     #OBSERVER: [3, (255, 255, 255)],
                    }

        for unit_type in draw_dict:
            for unit in self.units(unit_type).ready:
                pos = unit.position
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), draw_dict[unit_type][0], draw_dict[unit_type][1], -1)



        main_base_names = ["nexus", "commandcenter", "hatchery"]
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() not in main_base_names:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 5, (200, 50, 212), -1)
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() in main_base_names:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 15, (0, 0, 255), -1)

        for enemy_unit in self.known_enemy_units:

            if not enemy_unit.is_structure:
                worker_names = ["probe",
                                "scv",
                                "drone"]
                # if that unit is a PROBE, SCV, or DRONE... it's a worker
                pos = enemy_unit.position
                if enemy_unit.name.lower() in worker_names:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (55, 0, 155), -1)
                else:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 3, (50, 0, 215), -1)

        for obs in self.units(OBSERVER).ready:
            pos = obs.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (255, 255, 255), -1)

        # flip horizontally to make our final fix in visual representation:
        self.flipped = cv2.flip(game_data, 0)
        resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)

        cv2.imshow('Intel', resized)
        cv2.waitKey(1)

        if not HEADLESS:
            resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)
            cv2.imshow('Intel', resized)
            cv2.waitKey(1)

    async def build_workers(self):
        if (len(self.units(NEXUS)) * 22) > len(self.units(PROBE)) and len(self.units(PROBE)) < (self.MAX_WORKERS - (50/(self.times/4+1))):
            for nexus in self.units(NEXUS).ready.noqueue:
                if self.can_afford(PROBE):
                    await self.do(nexus.train(PROBE))
        for booster in self.units(NEXUS).ready:
            if booster.energy >= 50:
                for building in self.units.of_type([UnitTypeId.STARGATE, UnitTypeId.NEXUS, UnitTypeId.GATEWAY]).ready:
                    if not building.noqueue and not building.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                        await self.do(booster(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, building))
            

    async def build_pylons(self):
        if self.supply_left < 5 and not self.already_pending(PYLON):
            nexuses = self.units(NEXUS).ready
            if nexuses.exists:
                if self.can_afford(PYLON):
                    await self.build(PYLON, near=nexuses.first)

    async def build_assimilators(self):
        for nexus in self.units(NEXUS).ready:
            vaspenes = self.state.vespene_geyser.closer_than(15.0, nexus)
            for vaspene in vaspenes:
                if not self.can_afford(ASSIMILATOR):
                    break
                worker = self.select_build_worker(vaspene.position)
                if worker is None:
                    break
                if not self.units(ASSIMILATOR).closer_than(1.0, vaspene).exists:
                    await self.do(worker.build(ASSIMILATOR, vaspene))

    async def expand(self):
        if self.units(NEXUS).amount*22 < (len(self.units(PROBE))+6) and self.can_afford(NEXUS):
            await self.expand_now()

    async def offensive_force_buildings(self):
        #print(self.iteration / self.ITERATIONS_PER_MINUTE)
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random

            if self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE):
                if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE, near=pylon)

            elif len(self.units(GATEWAY).idle) < 1:
                if self.can_afford(GATEWAY) and self.units(GATEWAY).amount < self.units(NEXUS).amount*2 and not self.already_pending(GATEWAY):
                    await self.build(GATEWAY, near=pylon)

            if self.units(CYBERNETICSCORE).ready.exists:
                if len(self.units(ROBOTICSFACILITY)) < 1:
                    if self.can_afford(ROBOTICSFACILITY) and not self.already_pending(ROBOTICSFACILITY):
                        await self.build(ROBOTICSFACILITY, near=pylon)

            if self.units(CYBERNETICSCORE).ready.exists:
                if len(self.units(STARGATE).idle) < 1:
                    if self.can_afford(STARGATE) and not self.already_pending(STARGATE):
                        await self.build(STARGATE, near=pylon)

    async def build_offensive_force(self):
        for sg in self.units(STARGATE).ready.noqueue:
            if self.can_afford(VOIDRAY) and self.supply_left > 0 and self.units(NEXUS).amount*22 > (len(self.units(PROBE))+6):
                await self.do(sg.train(VOIDRAY))
        for gw in self.units(GATEWAY).ready.noqueue:
            if self.can_afford(ZEALOT) and self.supply_left > 0 and self.units(NEXUS).amount*22 > (len(self.units(PROBE))+6):
                await self.do(gw.train(ZEALOT))

    def find_target(self, state):
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units).position
        elif len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures).position
        else:
            return self.enemy_start_locations[0]


    async def attack(self):
        # {UNIT: [n to fight, n to defend]}
        aggressive_units = {VOIDRAY: [4, 12], ZEALOT: [4, 12]}
        army_units = self.units & []
        army_units += self.units(ZEALOT)
        army_units += self.units(VOIDRAY)
	
        for UNIT in aggressive_units:
            if self.units(UNIT).amount > aggressive_units[UNIT][0] and self.units(UNIT).amount > aggressive_units[UNIT][1]:
                for s in army_units.idle:
                    await self.do(s.attack(self.find_target(self.state)))

            elif self.units(UNIT).amount > aggressive_units[UNIT][1]:
                if len(self.known_enemy_units) > 0:
                    for s in army_units.idle:
                        await self.do(s.attack(random.choice(self.known_enemy_units).position))
                else: 
                    for s in army_units.idle:
                        await self.do(s.attack(random.choice(self.units(NEXUS)).position))

    async def activate_voidrays(self):
        for vr in self.units(VOIDRAY):
            abilities = await self.get_available_abilities(vr)
#            print(abilities)
            for unit in self.known_enemy_units:
                if AbilityId.EFFECT_VOIDRAYPRISMATICALIGNMENT in abilities and unit.position.to2.distance_to(vr.position.to2) < 7:
                    await self.do(vr(EFFECT_VOIDRAYPRISMATICALIGNMENT))
    
						
                    


				
run_game(maps.get("AbyssalReefLE"), [
    Bot(Race.Protoss, SentdeBot()),
    Computer(Race.Terran, Difficulty.Medium)
    ], realtime=False)