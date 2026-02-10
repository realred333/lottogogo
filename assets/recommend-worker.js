"use strict";

const NUMBERS = Array.from({ length: 45 }, (_, index) => index + 1);

let model = null;
let rawScores = [];
let boostTagsByNumber = [];
let exactHistorySet = new Set();
let fiveSubsetHistorySet = new Set();
let carryoverSet = new Set();
let rarePairs = [];
let initialized = false;

self.onmessage = (event) => {
  const message = event.data || {};
  const requestId = message.requestId;

  try {
    if (message.type === "init") {
      initModel(message.model);
      self.postMessage({ type: "inited", requestId });
      return;
    }

    if (message.type === "generate") {
      ensureInitialized();
      const payload = generateRecommendations(message);
      self.postMessage({ type: "result", requestId, payload });
      return;
    }

    throw new Error(`Unsupported worker message type: ${String(message.type)}`);
  } catch (error) {
    self.postMessage({
      type: "error",
      requestId,
      message: error instanceof Error ? error.message : "worker error",
    });
  }
};

function initModel(input) {
  if (!input || typeof input !== "object") {
    throw new Error("model.json payload is invalid");
  }

  model = input;
  rawScores = Array.isArray(model.raw_scores) ? model.raw_scores.map((value) => Number(value) || 0) : [];
  if (rawScores.length !== 45) {
    throw new Error("model.raw_scores must include exactly 45 numbers");
  }

  boostTagsByNumber = Array.isArray(model.boost_tags_by_number) ? model.boost_tags_by_number : [];
  if (boostTagsByNumber.length !== 45) {
    boostTagsByNumber = Array.from({ length: 45 }, () => []);
  }

  const history = model.history || {};
  exactHistorySet = new Set(Array.isArray(history.exact_keys) ? history.exact_keys : []);
  fiveSubsetHistorySet = new Set(Array.isArray(history.five_subset_keys) ? history.five_subset_keys : []);

  carryoverSet = new Set(Array.isArray(model.carryover_numbers) ? model.carryover_numbers.map((n) => Number(n)) : []);
  rarePairs = Array.isArray(model.rare_pairs)
    ? model.rare_pairs
        .map((pair) => [Number(pair?.[0]), Number(pair?.[1])])
        .filter((pair) => Number.isFinite(pair[0]) && Number.isFinite(pair[1]))
    : [];

  initialized = true;
}

function ensureInitialized() {
  if (!initialized || !model) {
    throw new Error("worker is not initialized");
  }
}

function generateRecommendations({ preset, games, recentKeys }) {
  const presetName = String(preset || "A");
  const gameCount = Number(games || 5);

  if (presetName !== "A" && presetName !== "B") {
    throw new Error("preset은 A 또는 B만 가능합니다");
  }
  if (gameCount !== 5 && gameCount !== 10) {
    throw new Error("games는 5 또는 10만 가능합니다");
  }

  const presetConfig = model.presets?.[presetName];
  if (!presetConfig) {
    throw new Error(`model preset config missing: ${presetName}`);
  }

  const sampling = presetConfig.sampling || {};
  const ranking = presetConfig.ranking || {};
  const filters = presetConfig.filters || {};
  const special = presetConfig.special || {};
  const reasons = Array.isArray(presetConfig.reasons) ? presetConfig.reasons : [];

  const weights = Array.isArray(sampling.weights) ? sampling.weights.map((value) => Number(value) || 0) : [];
  if (weights.length !== 45) {
    throw new Error(`model preset weights must include 45 numbers: ${presetName}`);
  }

  const maxAttempts = Number(sampling.max_attempts) > 0 ? Number(sampling.max_attempts) : 20000;
  const targetCandidates = Math.max(350, gameCount * 70);
  const recentSet = new Set(Array.isArray(recentKeys) ? recentKeys.map((key) => String(key)) : []);

  const candidatesByKey = new Map();
  let attempts = 0;

  while (attempts < maxAttempts && candidatesByKey.size < targetCandidates) {
    attempts += 1;
    const numbers = sampleWeightedCombination(weights);
    const key = comboKey(numbers);

    if (candidatesByKey.has(key) || recentSet.has(key)) {
      continue;
    }

    if (!passesAllFilters(numbers, key, filters, special)) {
      continue;
    }

    candidatesByKey.set(key, {
      numbers,
      score: scoreCombination(numbers),
      tags: buildTags(numbers),
      reasons,
      key,
    });
  }

  if (candidatesByKey.size < gameCount) {
    let relaxedAttempts = 0;
    const relaxedLimit = Math.max(6000, Math.floor(maxAttempts * 0.35));
    while (candidatesByKey.size < gameCount * 8 && relaxedAttempts < relaxedLimit) {
      relaxedAttempts += 1;
      const numbers = sampleWeightedCombination(weights);
      const key = comboKey(numbers);

      if (candidatesByKey.has(key) || recentSet.has(key) || !passesHistoryFilter(numbers, key)) {
        continue;
      }

      candidatesByKey.set(key, {
        numbers,
        score: scoreCombination(numbers),
        tags: buildTags(numbers),
        reasons,
        key,
      });
    }

    attempts += relaxedAttempts;
  }

  const ranked = [...candidatesByKey.values()].sort((left, right) => right.score - left.score);
  const selected = selectWithDiversity(ranked, gameCount, Number(ranking.max_overlap) || 3);

  if (!selected.length) {
    throw new Error("추천 가능한 조합을 생성하지 못했습니다. 잠시 후 다시 시도해주세요.");
  }

  const percentile = calculatePercentile(selected, ranked, Number(ranking.percentile_bias) || 0);

  return {
    meta: {
      preset: presetName,
      percentile,
    },
    recommendations: selected.map((item) => ({
      numbers: item.numbers,
      score: roundScore(item.score),
      tags: item.tags,
      reasons: item.reasons,
    })),
    debug: {
      attempts,
      candidateCount: ranked.length,
      generatedAt: new Date().toISOString(),
    },
  };
}

function sampleWeightedCombination(weights) {
  const localNumbers = NUMBERS.slice();
  const localWeights = weights.slice();
  const selected = [];

  for (let pick = 0; pick < 6; pick += 1) {
    const index = weightedPickIndex(localWeights);
    selected.push(localNumbers[index]);
    localNumbers.splice(index, 1);
    localWeights.splice(index, 1);
  }

  selected.sort((a, b) => a - b);
  return selected;
}

function weightedPickIndex(weights) {
  let total = 0;
  for (let i = 0; i < weights.length; i += 1) {
    total += Math.max(0, Number(weights[i]) || 0);
  }

  if (total <= 0) {
    return Math.floor(Math.random() * weights.length);
  }

  let target = Math.random() * total;
  for (let i = 0; i < weights.length; i += 1) {
    target -= Math.max(0, Number(weights[i]) || 0);
    if (target <= 0) {
      return i;
    }
  }

  return weights.length - 1;
}

function passesAllFilters(numbers, key, filters, special) {
  if (violatesBasicRanges(numbers, filters)) return false;
  if (violatesAC(numbers, filters)) return false;
  if (violatesZone(numbers, filters)) return false;
  if (violatesTail(numbers, filters)) return false;
  if (violatesSpecialRules(numbers, special)) return false;
  if (!passesHistoryFilter(numbers, key)) return false;
  return true;
}

function violatesBasicRanges(numbers, filters) {
  const minSum = Number(filters.min_sum);
  const maxSum = Number(filters.max_sum);
  const minOdd = Number(filters.min_odd);
  const maxOdd = Number(filters.max_odd);
  const minHigh = Number(filters.min_high);
  const maxHigh = Number(filters.max_high);

  const sum = numbers.reduce((acc, value) => acc + value, 0);
  if (Number.isFinite(minSum) && sum < minSum) return true;
  if (Number.isFinite(maxSum) && sum > maxSum) return true;

  const oddCount = numbers.reduce((acc, value) => acc + (value % 2 === 1 ? 1 : 0), 0);
  if (Number.isFinite(minOdd) && oddCount < minOdd) return true;
  if (Number.isFinite(maxOdd) && oddCount > maxOdd) return true;

  const highCount = numbers.reduce((acc, value) => acc + (value >= 23 ? 1 : 0), 0);
  if (Number.isFinite(minHigh) && highCount < minHigh) return true;
  if (Number.isFinite(maxHigh) && highCount > maxHigh) return true;

  return false;
}

function violatesAC(numbers, filters) {
  const minAc = Number(filters.min_ac);
  if (!Number.isFinite(minAc)) return false;

  const differences = new Set();
  for (let i = 0; i < numbers.length; i += 1) {
    for (let j = i + 1; j < numbers.length; j += 1) {
      differences.add(numbers[j] - numbers[i]);
    }
  }

  const acValue = differences.size - 5;
  return acValue < minAc;
}

function violatesZone(numbers, filters) {
  const maxPerZone = Number(filters.max_per_zone);
  if (!Number.isFinite(maxPerZone)) return false;

  const zoneCounts = [0, 0, 0, 0];
  for (const number of numbers) {
    if (number <= 11) zoneCounts[0] += 1;
    else if (number <= 22) zoneCounts[1] += 1;
    else if (number <= 33) zoneCounts[2] += 1;
    else zoneCounts[3] += 1;
  }

  return zoneCounts.some((count) => count > maxPerZone);
}

function violatesTail(numbers, filters) {
  const maxSameTail = Number(filters.max_same_tail);
  if (!Number.isFinite(maxSameTail)) return false;

  const tailCounts = new Map();
  for (const number of numbers) {
    const tail = number % 10;
    tailCounts.set(tail, (tailCounts.get(tail) || 0) + 1);
    if ((tailCounts.get(tail) || 0) > maxSameTail) {
      return true;
    }
  }

  return false;
}

function violatesSpecialRules(numbers, special) {
  if (Array.isArray(special.excluded_numbers) && special.excluded_numbers.length > 0) {
    const excluded = new Set(special.excluded_numbers.map((value) => Number(value)));
    for (const number of numbers) {
      if (excluded.has(number)) return true;
    }
  }

  if (special.rare_pair_filter) {
    const numberSet = new Set(numbers);
    for (const pair of rarePairs) {
      if (numberSet.has(pair[0]) && numberSet.has(pair[1])) {
        return true;
      }
    }
  }

  const maxCarryover = special.max_carryover_in_combo;
  if (Number.isFinite(Number(maxCarryover))) {
    let carryoverCount = 0;
    for (const number of numbers) {
      if (carryoverSet.has(number)) {
        carryoverCount += 1;
        if (carryoverCount > Number(maxCarryover)) return true;
      }
    }
  }

  return false;
}

function passesHistoryFilter(numbers, key) {
  const matchThreshold = Number(model?.history?.match_threshold || 5);

  if (matchThreshold >= 6) {
    return !exactHistorySet.has(key);
  }

  if (matchThreshold === 5) {
    if (exactHistorySet.has(key)) return false;

    for (let skip = 0; skip < 6; skip += 1) {
      const subset = [];
      for (let index = 0; index < 6; index += 1) {
        if (index !== skip) subset.push(numbers[index]);
      }
      if (fiveSubsetHistorySet.has(comboKey(subset))) {
        return false;
      }
    }

    return true;
  }

  return !exactHistorySet.has(key);
}

function scoreCombination(numbers) {
  let score = 0;
  for (const number of numbers) {
    score += Number(rawScores[number - 1] || 0);
  }
  return score;
}

function buildTags(numbers) {
  const counts = new Map();

  for (const number of numbers) {
    const tags = Array.isArray(boostTagsByNumber[number - 1]) ? boostTagsByNumber[number - 1] : [];
    for (const tag of tags) {
      counts.set(tag, (counts.get(tag) || 0) + 1);
    }
  }

  if (!counts.size) {
    return ["pattern"];
  }

  return [...counts.entries()]
    .sort((left, right) => {
      if (right[1] !== left[1]) return right[1] - left[1];
      return String(left[0]).localeCompare(String(right[0]));
    })
    .slice(0, 3)
    .map((entry) => String(entry[0]));
}

function selectWithDiversity(ranked, games, maxOverlap) {
  const selected = [];
  const selectedKeySet = new Set();

  for (const item of ranked) {
    if (violatesOverlap(item.numbers, selected, maxOverlap)) continue;

    selected.push(item);
    selectedKeySet.add(item.key);
    if (selected.length >= games) break;
  }

  if (selected.length < games) {
    for (const item of ranked) {
      if (selectedKeySet.has(item.key)) continue;
      selected.push(item);
      selectedKeySet.add(item.key);
      if (selected.length >= games) break;
    }
  }

  return selected;
}

function violatesOverlap(candidate, selected, maxOverlap) {
  const candidateSet = new Set(candidate);

  for (const item of selected) {
    let overlap = 0;
    for (const number of item.numbers) {
      if (candidateSet.has(number)) {
        overlap += 1;
        if (overlap >= maxOverlap + 1) {
          return true;
        }
      }
    }
  }

  return false;
}

function calculatePercentile(selected, ranked, percentileBias) {
  if (!selected.length || !ranked.length) {
    return null;
  }

  const rankMap = new Map();
  for (let index = 0; index < ranked.length; index += 1) {
    rankMap.set(ranked[index].key, index + 1);
  }

  const values = [];
  for (const item of selected) {
    const rank = rankMap.get(item.key);
    if (!rank) continue;
    values.push(Math.max(1, Math.min(100, Math.round((rank / ranked.length) * 100))));
  }

  if (!values.length) {
    return null;
  }

  const average = values.reduce((acc, value) => acc + value, 0) / values.length;
  const adjusted = Math.round(average) + percentileBias;
  return Math.max(1, Math.min(100, adjusted));
}

function comboKey(numbers) {
  return numbers.join("-");
}

function roundScore(value) {
  return Math.round(Number(value) * 1_000_000) / 1_000_000;
}
