import math
import random
from collections import OrderedDict, defaultdict
from typing import Dict, List, Set

import numpy as np
from luxai_s2.actions import DigAction, MoveAction, SelfDestructAction
from luxai_s2.env import (ActionsByType, LuxAI_S2, Position, TransferAction, move_deltas, resource_to_name)
from luxai_s2.map.board import Board
from luxai_s2.state import State
from luxai_s2.state.stats import (
    GenerationStatsStateDict,
    StatsStateDict,
    create_all_stats,
    create_consumption_stats,
    create_destroyed_stats,
    create_generation_stats,
    create_robot_stats,
    create_transfer_pickup_stats,
)
from luxai_s2.unit import Unit, UnitType
from luxai_s2.utils.utils import get_top_two_power_units


def create_transfer_stats():
    return dict(
        power=create_all_stats(),
        water=create_all_stats(),
        metal=create_all_stats(),
        ice=create_all_stats(),
        ore=create_all_stats(),
    )


def create_destroyed_stats():
    return dict(
        FACTORY=0,
        HEAVY=dict(own=0, enm=0),
        LIGHT=dict(own=0, enm=0),
        rubble=create_robot_stats(),
        lichen=dict(
            LIGHT=dict(own=0, enm=0),
            HEAVY=dict(own=0, enm=0),
        ),
    )


def create_empty_stats() -> StatsStateDict:
    stats: StatsStateDict = dict()
    stats["action_queue_updates_total"] = 0
    stats["action_queue_updates_success"] = 0
    stats["consumption"] = create_consumption_stats()
    stats["destroyed"] = create_destroyed_stats()
    stats["generation"] = create_generation_stats()
    stats["pickup"] = create_transfer_pickup_stats()
    stats["transfer"] = create_transfer_stats()
    return stats


def reset(self, seed=None):
    """
    Reset needs to initialize the `agents` attribute and must set up the
    environment so that render(), and step() can be called without issues.

    Here it initializes the `num_moves` variable which counts the number of
    hands that are played.

    Returns the observations for each agent
    """

    self.agents = self.possible_agents[:]
    self.env_steps = 0
    if seed is not None:
        self.seed_val = seed
        self.seed_rng = np.random.RandomState(seed=seed)
    else:
        self.seed_val = np.random.randint(0, 2**32 - 1)
        self.seed_rng = np.random.RandomState(seed=self.seed_val)
    board = Board(seed=self.seed_rng.randint(0, 2**32 - 1, dtype=np.int64), env_cfg=self.env_cfg)
    self.state: State = State(
        seed_rng=self.seed_rng,
        seed=self.seed_val,
        env_cfg=self.state.env_cfg,
        env_steps=0,
        board=board,
    )
    self.max_episode_length = self.env_cfg.max_episode_length
    for agent in self.possible_agents:
        self.state.units[agent] = OrderedDict()
        self.state.factories[agent] = OrderedDict()
        if self.collect_stats:
            self.state.stats[agent] = create_empty_stats()
    obs = self.state.get_obs()
    observations = {agent: obs for agent in self.agents}
    return observations


def _handle_transfer_actions(self, actions_by_type: ActionsByType):
    # It is important to first sub resource from all units, and then add
    # resource to targets. Only When splitted into two loops, the transfer
    # action is irrelevant to unit id.

    # sub from unit cargo
    amount_list = []
    for unit, transfer_action in actions_by_type["transfer"]:
        transfer_action: TransferAction
        transfer_amount = unit.sub_resource(transfer_action.resource, transfer_action.transfer_amount)
        amount_list.append(transfer_amount)

    # add to target cargo
    for (unit, transfer_action), transfer_amount in zip(actions_by_type["transfer"], amount_list):
        transfer_action: TransferAction
        transfer_pos: Position = (unit.pos + move_deltas[transfer_action.transfer_dir])
        units_there = self.state.board.get_units_at(transfer_pos)
        # if there is a factory, we prefer transferring to that entity
        factory_id = f"factory_{self.state.board.factory_occupancy_map[transfer_pos.x, transfer_pos.y]}"
        if factory_id in self.state.factories[unit.team.agent]:
            factory = self.state.factories[unit.team.agent][factory_id]
            actually_transferred = factory.add_resource(transfer_action.resource, transfer_amount)
            if self.collect_stats:
                self.state.stats[unit.team.agent]["transfer"][resource_to_name[
                    transfer_action.resource]]["FACTORY"] += actually_transferred
        elif units_there is not None:
            assert len(units_there) == 1, "Fatal error here, this is a bug"
            target_unit: Unit = units_there[0]
            # add resources to target. This will waste (transfer_amount - actually_transferred) resources
            actually_transferred = target_unit.add_resource(transfer_action.resource, transfer_amount)
            if self.collect_stats:
                if target_unit.unit_type == UnitType.LIGHT:
                    self.state.stats[unit.team.agent]["transfer"][resource_to_name[
                        transfer_action.resource]]["LIGHT"] += actually_transferred
                elif target_unit.unit_type == UnitType.HEAVY:
                    self.state.stats[unit.team.agent]["transfer"][resource_to_name[
                        transfer_action.resource]]["HEAVY"] += actually_transferred
                else:
                    raise NotImplemented
        unit.repeat_action(transfer_action)


old_handle_bid = LuxAI_S2._handle_bid


def _handle_bid(self, actions):
    rval = old_handle_bid(self, actions)
    if not self.load_from_replay:
        if actions['player_0']['bid'] == actions['player_1']['bid']:
            rand = random.random()
            self.state.teams['player_0'].place_first = rand > 0.5
            self.state.teams['player_1'].place_first = not (rand > 0.5)
    return rval


def _handle_dig_actions(self: LuxAI_S2, actions_by_type: ActionsByType):
    for unit, dig_action in actions_by_type["dig"]:
        dig_action: DigAction
        if self.state.board.rubble[unit.pos.x, unit.pos.y] > 0:
            if self.collect_stats:
                rubble_before = self.state.board.rubble[unit.pos.x, unit.pos.y]
            self.state.board.rubble[unit.pos.x, unit.pos.y] = max(
                self.state.board.rubble[unit.pos.x, unit.pos.y] - unit.unit_cfg.DIG_RUBBLE_REMOVED,
                0,
            )
            if self.collect_stats:
                self.state.stats[unit.team.agent]["destroyed"]["rubble"][unit.unit_type.name] -= (
                    self.state.board.rubble[unit.pos.x, unit.pos.y] - rubble_before)
        elif self.state.board.lichen[unit.pos.x, unit.pos.y] > 0:
            if self.collect_stats:
                lichen_before = self.state.board.lichen[unit.pos.x, unit.pos.y]
            lichen_left = max(
                self.state.board.lichen[unit.pos.x, unit.pos.y] - unit.unit_cfg.DIG_LICHEN_REMOVED,
                0,
            )
            self.state.board.lichen[unit.pos.x, unit.pos.y] = lichen_left
            if lichen_left == 0:  # dug out the last lichen
                self.state.board.rubble[unit.pos.x,
                                        unit.pos.y] = self.state.env_cfg.ROBOTS[unit.unit_type.name].DIG_RESOURCE_GAIN
            if self.collect_stats:
                strain_id = self.state.board.lichen_strains[unit.pos.x, unit.pos.y]
                own = False
                for f in self.state.factories[unit.team.agent].values():
                    if f.num_id == strain_id:
                        own = True
                        break
                ownner = "own" if own else "enm"
                self.state.stats[unit.team.agent]["destroyed"]["lichen"][unit.unit_type.name][ownner] -= (
                    self.state.board.lichen[unit.pos.x, unit.pos.y] - lichen_before)
        elif self.state.board.ice[unit.pos.x, unit.pos.y] > 0:
            gained = unit.add_resource(0, unit.unit_cfg.DIG_RESOURCE_GAIN)
            if self.collect_stats:
                self.state.stats[unit.team.agent]["generation"]["ice"][unit.unit_type.name] += gained
        elif self.state.board.ore[unit.pos.x, unit.pos.y] > 0:
            gained = unit.add_resource(1, unit.unit_cfg.DIG_RESOURCE_GAIN)
            if self.collect_stats:
                self.state.stats[unit.team.agent]["generation"]["ore"][unit.unit_type.name] += gained
        unit.power -= self.state.env_cfg.ROBOTS[unit.unit_type.name].DIG_COST
        unit.repeat_action(dig_action)


def _handle_movement_actions(self: LuxAI_S2, actions_by_type: ActionsByType):
    new_units_map: Dict[str, List[Unit]] = defaultdict(list)
    heavy_entered_pos: Dict[str, List[Unit]] = defaultdict(list)
    light_entered_pos: Dict[str, List[Unit]] = defaultdict(list)

    for unit, move_action in actions_by_type["move"]:
        move_action: MoveAction
        # skip move center
        if move_action.move_dir != 0:
            old_pos_hash = self.state.board.pos_hash(unit.pos)
            target_pos = (unit.pos + move_action.dist * move_deltas[move_action.move_dir])
            power_required = move_action.power_cost
            unit.pos = target_pos
            new_pos_hash = self.state.board.pos_hash(unit.pos)

            # remove unit from map temporarily
            if len(self.state.board.units_map[old_pos_hash]) == 1:
                del self.state.board.units_map[old_pos_hash]
            else:
                self.state.board.units_map[old_pos_hash].remove(unit)

            new_units_map[new_pos_hash].append(unit)
            unit.power -= power_required

            if unit.unit_type == UnitType.HEAVY:
                heavy_entered_pos[new_pos_hash].append(unit)
            else:
                light_entered_pos[new_pos_hash].append(unit)

        unit.repeat_action(move_action)

    for pos_hash, units in self.state.board.units_map.items():
        # add in all the stationary units
        new_units_map[pos_hash] += units

    all_destroyed_units: Set[Unit] = set()
    new_units_map_after_collision: Dict[str, List[Unit]] = defaultdict(list)
    for pos_hash, units in new_units_map.items():
        destroyed_units: Set[Unit] = set()
        if len(units) <= 1:
            new_units_map_after_collision[pos_hash] += units
            continue
        if len(heavy_entered_pos[pos_hash]) > 1:
            # all units collide, find the top 2 units by power
            (most_power_unit, next_most_power_unit) = get_top_two_power_units(units, UnitType.HEAVY)
            if most_power_unit.power == next_most_power_unit.power:
                # tie, all units break
                for u in units:
                    destroyed_units.add(u)
                self._log(
                    f"{len(destroyed_units)} Units: ({', '.join([u.unit_id for u in destroyed_units])}) collided at {pos_hash}"
                )
            else:
                most_power_unit_power_loss = math.ceil(next_most_power_unit.power * self.env_cfg.POWER_LOSS_FACTOR)
                most_power_unit.power -= most_power_unit_power_loss
                surviving_unit = most_power_unit
                for u in units:
                    if u.unit_id != surviving_unit.unit_id:
                        destroyed_units.add(u)
                self._log(
                    f"{len(destroyed_units)} Units: ({', '.join([u.unit_id for u in destroyed_units])}) collided at {pos_hash} with {surviving_unit} surviving with {surviving_unit.power} power"
                )
                new_units_map_after_collision[pos_hash].append(surviving_unit)
            all_destroyed_units.update(destroyed_units)
        elif len(heavy_entered_pos[pos_hash]) > 0:
            # all other units collide and break
            surviving_unit = heavy_entered_pos[pos_hash][0]
            for u in units:
                if u.unit_id != surviving_unit.unit_id:
                    destroyed_units.add(u)
            self._log(
                f"{len(destroyed_units)} Units: ({', '.join([u.unit_id for u in destroyed_units])}) collided at {pos_hash} with {surviving_unit} surviving with {surviving_unit.power} power"
            )
            new_units_map_after_collision[pos_hash].append(surviving_unit)
            all_destroyed_units.update(destroyed_units)
        else:
            # check for stationary heavy unit there
            surviving_unit = None
            heavy_stationary_unit = None
            for u in units:
                if u.unit_type == UnitType.HEAVY:
                    if heavy_stationary_unit is not None:
                        heavy_stationary_unit = None
                        # we found >= 2 heavies stationary in a tile where no heavies are entering.
                        # should only happen when spawning units
                        self._log(f"At {pos_hash}, >= 2 heavies crashed as they were all stationary")
                        break
                    heavy_stationary_unit = u

            if heavy_stationary_unit is not None:
                surviving_unit = heavy_stationary_unit
            else:
                if len(light_entered_pos[pos_hash]) > 1:
                    # all units collide, get top 2 units by power
                    (
                        most_power_unit,
                        next_most_power_unit,
                    ) = get_top_two_power_units(units, UnitType.LIGHT)
                    if most_power_unit.power == next_most_power_unit.power:
                        # tie, all units break
                        for u in units:
                            destroyed_units.add(u)
                    else:
                        most_power_unit_power_loss = math.ceil(next_most_power_unit.power *
                                                               self.env_cfg.POWER_LOSS_FACTOR)
                        most_power_unit.power -= most_power_unit_power_loss
                        surviving_unit = most_power_unit
                elif len(light_entered_pos[pos_hash]) > 0:
                    # light crashes into stationary light unit
                    surviving_unit = light_entered_pos[pos_hash][0]
            if surviving_unit is None:
                for u in units:
                    destroyed_units.add(u)
                self._log(
                    f"{len(destroyed_units)} Units: ({', '.join([u.unit_id for u in destroyed_units])}) collided at {pos_hash}"
                )
                all_destroyed_units.update(destroyed_units)
            else:
                for u in units:
                    if u.unit_id != surviving_unit.unit_id:
                        destroyed_units.add(u)
                self._log(
                    f"{len(destroyed_units)} Units: ({', '.join([u.unit_id for u in destroyed_units])}) collided at {pos_hash} with {surviving_unit} surviving with {surviving_unit.power} power"
                )
                new_units_map_after_collision[pos_hash].append(surviving_unit)
                all_destroyed_units.update(destroyed_units)
    self.state.board.units_map = new_units_map_after_collision

    for u in all_destroyed_units:
        self.destroy_unit(u)

    if self.collect_stats:
        for u in all_destroyed_units:
            pos_hash = self.state.board.pos_hash(u.pos)
            surviving = new_units_map_after_collision[pos_hash]
            if surviving:
                surviving = surviving[0]
                own_or_enm = 'own' if surviving.team.agent == u.team.agent else 'enm'
                self.state.stats[surviving.team.agent]["destroyed"][u.unit_type.name][own_or_enm] += 1


def _handle_self_destruct_actions(self, actions_by_type: ActionsByType):
    for unit, self_destruct_action in actions_by_type["self_destruct"]:
        unit: Unit
        self_destruct_action: SelfDestructAction
        pos_hash = self.state.board.pos_hash(unit.pos)
        del self.state.board.units_map[pos_hash]
        self.destroy_unit(unit)
        if self.collect_stats:
            self.state.stats[unit.team.agent]["destroyed"][unit.unit_type.name]['own'] += 1


def install_patch():
    LuxAI_S2.reset = reset
    LuxAI_S2._handle_transfer_actions = _handle_transfer_actions
    LuxAI_S2._handle_bid = _handle_bid
    LuxAI_S2._handle_dig_actions = _handle_dig_actions
    LuxAI_S2._handle_movement_actions = _handle_movement_actions
    LuxAI_S2._handle_self_destruct_actions = _handle_self_destruct_actions
