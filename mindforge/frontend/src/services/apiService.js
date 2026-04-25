import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL,
});

export const getClarificationQuestions = async (text) => {
  try {
    const response = await api.post('/clarify', { text });
    return response.data;
  } catch (error) {
    console.error('Error getting clarification questions:', error);
    throw error;
  }
};

export const generateMindMap = async (text, clarificationAnswers = null, additionalInfo = "") => {
  try {
    const response = await api.post('/generate', {
      text,
      clarification_answers: clarificationAnswers,
      additional_info: additionalInfo
    });
    return response.data;
  } catch (error) {
    console.error('Error generating mind map:', error);
    throw error;
  }
};

export const refineMindMap = async (previousMap, feedback) => {
  try {
    const response = await api.post('/refine', {
      previous_map: previousMap,
      feedback: feedback
    });
    return response.data;
  } catch (error) {
    console.error('Error refining mind map:', error);
    throw error;
  }
};
